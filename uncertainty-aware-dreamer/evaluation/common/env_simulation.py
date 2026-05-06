import torch


def get_env_infos_from_actions(data_collector,
                               actions,
                               start_phys_state,
                               phys_state_is_obs):
    """
    Perform rollouts from the given actions and return the physical states.
    """
    observations, rewards, phys_states = [], [], []
    for i in range(actions.shape[0]):
        start_ps = start_phys_state if len(start_phys_state.shape) == 1 else start_phys_state[i]
        obs, rew, infos = data_collector.rollout_from_actions(pred_actions=actions[i],
                                                              start_phys_state=start_ps,
                                                              max_length=actions.shape[1]-1) # first phys state is included in length
        observations.append(obs)
        rewards.append(rew)
        phys_states.append(obs[1] if phys_state_is_obs else infos["state"])

    observations = [torch.stack([o[i] for o in observations], dim=0) for i in range(len(observations[0]))]
    rewards = torch.stack(rewards, dim=0)
    phys_states = torch.stack(phys_states, dim=0)
    
    return observations, rewards, phys_states


def get_env_obs_from_phys_states(data_collector, phys_states, normed=True):
    """
    Simulate physical state in environment, record observations.
    """
    obs = []
    for i in range(phys_states.shape[0]):
        obs_i = []
        for j in range(phys_states.shape[1]):
            o, _ = data_collector._env.reset_to_state(phys_states[i,j].cpu(), normed)
            obs_i.append(o[0])
        obs.append(torch.stack(obs_i, 0))
    return torch.stack(obs, 0)