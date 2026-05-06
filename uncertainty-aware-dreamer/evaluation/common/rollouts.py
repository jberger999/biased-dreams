import torch
import faiss

from uncertainty_aware_dreamer.ssm_mbrl.util.stack_util import stack_maybe_nested_dicts


@torch.no_grad()
def get_posteriors(data_collector,
                   start_model_state,
                   start_phys_state,
                   num_rollouts,
                   rollout_length,
                   pred_actions=None,
                   sample_model=True):
    """
    Perform posterior rollouts from the given actions.
    """
    posterior_states = []
    phys_states = []
    observations = []
    actions = []
    rewards = []
    
    if pred_actions is not None:
        # Rollout each individual sequence according to the predicted action sequence.
        for i in range(num_rollouts):
            # infos["state"] = [rollout_length+1 x #phys_dims]
            obs, r, infos, states = data_collector.posterior_rollout_from_actions(max_length=rollout_length,
                                                                                  pred_actions=pred_actions,
                                                                                  start_phys_state=start_phys_state,
                                                                                  start_model_state=start_model_state,
                                                                                  sample_model=sample_model) 
            posterior_states.append(states)
            phys_states.append(infos["state"])
            observations.append(obs)
            rewards.append(r)
            
        # Same action sequence for all rollouts.
        actions = pred_actions.unsqueeze(0).repeat(num_rollouts, 1, 1).to(posterior_states[0]["sample"].device)
    else:
        # Rollout each individual sequence according to policy.
        for i in range(num_rollouts):
            obs, act, r, infos, states = data_collector.rollout_policy(max_length=rollout_length,
                                                                       start_phys_state=start_phys_state,
                                                                       start_model_state=start_model_state,
                                                                       start_action=None,   # use 0 action for start
                                                                       sample_policy=False, # greedy actions
                                                                       sample_model=True)   # act upon sampled model states
            posterior_states.append(states)
            phys_states.append(infos["state"])
            observations.append(obs)
            actions.append(act)
            rewards.append(r)
            
        # Stack over all sequences.
        actions = torch.stack(actions, dim=0)

    posterior_states = stack_maybe_nested_dicts(posterior_states, dim=0)
    phys_states = torch.stack(phys_states, dim=0)
    observations_stacked = [torch.stack(obs, 0) for obs in zip(*observations)]  
    rewards = torch.stack(rewards, dim=0)

    return posterior_states, phys_states, observations_stacked, actions, rewards


@torch.no_grad()
def get_priors(data_collector,
               start_model_state,
               num_rollouts,
               include_first=True,
               sample_model=False,
               pred_actions=None):
    """
    Perform prior rollouts from the given actions or purely from imagination, when no actions are given.
    """
    # We can perform the rollouts in parallel.
    if start_model_state["gru_cell_state"].shape[0] == 1:
        # Only one start state given.
        start_model_state = data_collector._model.repeat_state(start_model_state, num_rollouts)
    if pred_actions is not None:
        if len(pred_actions.shape) == 2:
            # Only one action sequence given.
            actions = pred_actions.unsqueeze(0).repeat(num_rollouts, 1, 1).to(start_model_state["sample"].device)
        else:
            actions = pred_actions
        # prior_states include start_model_state_repeat
        prior_states = data_collector.prior_rollout_from_actions(pred_actions=actions,
                                                                 start_model_states=start_model_state,
                                                                 sample_model=sample_model,
                                                                 include_first=include_first) 
    else:
        # prior_states include start_model_state_repeat
        prior_states, _, actions = data_collector.imagine_rollout(start_model_states=start_model_state,
                                                                  sample_policy=False, # greedy actions
                                                                  sample_model=sample_model,
                                                                  include_first=include_first)
            
    return prior_states, actions


def get_one_step_priors(eval_data_collector, posterior_states, gt_actions, rollout_length, num_rollouts, open_loop):
    """
    Compute one-step priors with posterior background, which is the state before being updated with the
    corresponding observation.
    """
    # Enforce one-step imagination only.
    prev_imagine_horizon = eval_data_collector._imagine_horizon
    eval_data_collector._imagine_horizon = 1
    
    prior_posterior_states = []
    first_step = {k: v[:, 0] for k, v in posterior_states.items()}
    if posterior_states["gru_cell_state"].shape[0] == 1:  
        first_step = eval_data_collector._model.repeat_state(first_step[None], num_rollouts)  
    prior_posterior_states.append(first_step)
    
    for t in range(rollout_length - 1):
        start_state = {k: v[:, t] for k, v in posterior_states.items()}
        prior_states, _ = get_priors(data_collector=eval_data_collector,
                                     pred_actions=gt_actions[:, t] if open_loop else None,
                                     start_model_state=start_state,
                                     num_rollouts=num_rollouts,
                                     include_first=False,
                                     sample_model=False)
        prior_states = {k: v.squeeze(1) for k, v in prior_states.items()}
        prior_posterior_states.append(prior_states)
        
    eval_data_collector._imagine_horizon = prev_imagine_horizon 

    return stack_maybe_nested_dicts(prior_posterior_states, dim=1)


def get_start_state(data_collector, random=False, step_idx=None):
    """
    Determine start state, either random or specifically chosen.
    If random=False, then:
    1.  Collect some amount of episodes. From these episodes determine the most extreme state in any dimension.
    2.  Determine starting latent state from the physical state.
    """
    _, actions, _, infos, post_states = data_collector.collect(sample_policy=False, # greedy actions
                                                               sample_model=False)  # act upon deterministic model states

    # [num_init_episodes x episode_length+1 x #phys_dims]
    phys_states = torch.stack([info["state"] for info in infos], dim=0)
    
    if step_idx is not None:
        # If step index provied, use random episode, but specific step index
        assert step_idx < phys_states.shape[1]
        episode_idx = torch.randint(phys_states.shape[0], (1,)).squeeze(-1)
    elif random:
        episode_idx = torch.randint(phys_states.shape[0], (1,)).squeeze(-1)
        step_idx = torch.randint(phys_states.shape[1], (1,)).squeeze(-1)
    else:
        # Determine index of posterior states, that maximizes the physical state in any dimension.
        valid_phys_states = phys_states[:, 1:]
        
        max_dim_indices = torch.argmax(valid_phys_states, dim=-1, keepdim=True)
        max_dim = torch.take_along_dim(valid_phys_states, max_dim_indices, dim=-1).squeeze(-1)
        max_step_indices = torch.argmax(max_dim, dim=-1, keepdim=True)
        max_step = torch.take_along_dim(max_dim, max_step_indices, dim=-1).squeeze(-1)
        
        episode_idx = torch.argmax(max_step, dim=-1)
        step_idx = max_step_indices[episode_idx].item()

    # Start model state is a dictionary coinciding with the keys of post_states.
    # Entries are tensors of shape [1 x state_dim]
    start_model_state = {}
    for key in post_states[0]:
        start_model_state[key] = post_states[episode_idx][key][step_idx].unsqueeze(0)
    
    # Start physical state is the physical state at the determined index, which we want to reset the environment to.
    # step_idx - 1, since we have a index misalignment due to 0-th obs/info being appended before being associated with the 0-th model state
    # [1 x #phys_dims]
    start_phys_state = infos[episode_idx]["state"][step_idx - 1]

    # Get start action.
    start_action = actions[episode_idx][step_idx].unsqueeze(0)

    return start_model_state, start_phys_state, start_action


def get_most_id_state(dataset, frac=1, k=10):
    """
    Find state with smallest mean distance to its k-neighbors in the dataset.
    """
    num_seq, seq_len, dim = dataset.shape
    dataset = dataset.flatten(0, 1)
    
    # Use only fraction of dataset, since otherwise pairwise comparison takes way too long
    subset_idx = torch.randperm(num_seq * seq_len)[:int(num_seq * seq_len / frac)]
    data_subset = dataset[subset_idx]
    
    data_np = data_subset.cpu().numpy().astype("float32")

    index = faiss.IndexFlatL2(dim)
    index.add(data_np)
    
    dists, _ = index.search(data_np, k + 1) # k+1 because the first neighbor is the point itself

    mean_knn_dist = dists[:, 1:].mean(axis=1) # omit point itself
    best_idx = mean_knn_dist.argmin()
    best_mean_knn_dist = mean_knn_dist.min()
    best_point = data_subset[best_idx]
    
    return best_point, best_mean_knn_dist
    

def get_sequence(data_collector, sequence_length, random=False):
    """
    Collect entire sequence with given sequence length.
    """
    all_obs, actions, _, infos, post_states = data_collector.collect(sample_policy=False,   # greedy actions
                                                                     sample_model=False)    # act upon deterministic model states

    # [num_init_episodes x episode_length+1 x #phys_dims]
    phys_states = torch.stack([info["state"] for info in infos], dim=0)
    
    # We can only sample at most the episode length
    assert phys_states.shape[1] >= sequence_length
    
    if not random:
        valid_phys_states = phys_states[:, 1:-sequence_length]
        
        # Determine index of posterior states, that maximizes the physical state in any dimension.
        max_dim_indices = torch.argmax(valid_phys_states, dim=-1, keepdim=True)
        max_dim = torch.take_along_dim(valid_phys_states, max_dim_indices, dim=-1).squeeze(-1)
        max_step_indices = torch.argmax(max_dim, dim=-1, keepdim=True)
        max_step = torch.take_along_dim(max_dim, max_step_indices, dim=-1).squeeze(-1)
        
        episode_idx = torch.argmax(max_step, dim=-1)
        start_idx = max_step_indices[episode_idx].item()
    else:
        episode_idx = torch.randint(phys_states.shape[0], (1,)).squeeze(-1)
        start_idx = torch.randint(1, phys_states.shape[1] - sequence_length - 1, (1,)).squeeze(-1)
    
    # Sequence slice for model states and actions
    seq_slice = slice(start_idx, start_idx + sequence_length)
    # Sequence slice for observations and physical states, since we have a index misalignment due to 0-th obs/info being appended before being associated with the 0-th model state
    seq_slice_shift = slice(start_idx - 1, start_idx - 1 + sequence_length)

    # Start model state is a dictionary coinciding with the keys of post_states.
    # Entries are tensors of shape [1 x state_dim]
    model_states = {k: post_states[episode_idx][k][seq_slice].squeeze(1) for k in post_states[0]}
    
    # Start physical state is the physical state at the determined index, which we want to reset the environment to.
    # [1 x #phys_dims]
    phys_states = infos[episode_idx]["state"][seq_slice_shift]

    # Get start action.
    actions = actions[episode_idx][seq_slice]
    
    # Extract observation sequence
    obs = [all_obs[episode_idx][i][seq_slice_shift] for i in range(len(all_obs[0]))]

    return obs, model_states, phys_states, actions


def get_posterior_seq(data_collector, rollout_length):
    """
    Get one posterior sequence far into the episode.
    """
    start_model_state, start_phys_state, _ = get_start_state(data_collector=data_collector,
                                                             step_idx=200)
    # Posterior rollout from start conditions
    post_states, post_phys_states, post_obs, post_act, post_rew = get_posteriors(data_collector=data_collector,
                                                                                 start_model_state=start_model_state,
                                                                                 start_phys_state=start_phys_state,
                                                                                 num_rollouts=1,
                                                                                 rollout_length=rollout_length)
    return post_states, post_phys_states, post_obs, post_act, post_rew


def get_random_posteriors_priors(data_collector,
                                 rollout_length=100,
                                 num_searches=1000,
                                 verbose=True):
    """
    Collect random posterior and prior rollouts. Priors start at a random model state of the previously
    computed posterior rollout for efficiency reasons.
    """
    prev_imagine_horizon = data_collector._imagine_horizon
    prev_seq_per_collect = data_collector._sequences_per_collect
    data_collector._imagine_horizon = rollout_length
    data_collector._sequences_per_collect = 1
    
    all_post_states,  all_post_act, all_post_start_phys_states = [], [], []
    all_prior_states, all_prior_act, all_prior_start_phys_states = [], [], []
    for i in range(num_searches):
        # Sample random posterior sequence.
        _, post_states, post_phys_states, post_act = get_sequence(data_collector=data_collector,
                                                                  sequence_length=rollout_length,
                                                                  random=True)
        
        # Sample random prior sequence, starting from some random posterior state.
        rand_idx = torch.randint(0, post_phys_states.shape[0], (1,))
        start_model_state = {k: v[rand_idx, :] for k, v in post_states.items()}
        start_phys_state = post_phys_states[rand_idx, :]
        prior_states, prior_acts = get_priors(data_collector=data_collector,
                                              start_model_state=start_model_state,
                                              num_rollouts=1,
                                              sample_model=True)         # act upon sampled model states
        
        prior_states = {k: v.squeeze(0) for k, v in prior_states.items()}
        
        all_post_states.append(post_states)
        all_post_act.append(post_act)
        all_post_start_phys_states.append(post_phys_states[0])
        
        all_prior_states.append(prior_states)
        all_prior_act.append(prior_acts.squeeze(0))
        all_prior_start_phys_states.append(start_phys_state[0])
        
        if verbose:
            print("[COLLECT] Finished", i, "searches.")
        
    all_post_states = stack_maybe_nested_dicts(all_post_states, dim=0)
    all_post_act = torch.stack(all_post_act, dim=0)
    all_post_start_phys_states = torch.stack(all_post_start_phys_states, dim=0)
    
    all_prior_states = stack_maybe_nested_dicts(all_prior_states, dim=0)
    all_prior_act = torch.stack(all_prior_act, dim=0)
    all_prior_start_phys_states = torch.stack(all_prior_start_phys_states, dim=0)
    
    data_collector._imagine_horizon = prev_imagine_horizon
    data_collector._sequences_per_collect = prev_seq_per_collect
    
    all_dict = {
        "post_states": all_post_states,
        "post_act": all_post_act,
        "post_start_phys_state": all_post_start_phys_states,
        "prior_states": all_prior_states,
        "prior_act": all_prior_act,
        "prior_start_phys_state": all_prior_start_phys_states,
    }
    
    return all_dict