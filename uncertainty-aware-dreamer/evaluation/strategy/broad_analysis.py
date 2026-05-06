import torch
import os
import json
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
import random

from uncertainty_aware_dreamer.ssm_mbrl.common.img_preprocessor import ImgPreprocessor
from uncertainty_aware_dreamer.envs.env_factory import StateBasedDMCEnvFactory

from evaluation.utils.data_collector import EvalDataCollector
from evaluation.utils.saves import save_obs_trajectory, save_to_csv
from evaluation.utils.visuals import plot_funcs_shared, plot_vector_field_with_lines

from evaluation.common.decoding import get_uncertainties, get_reconstructions, get_decoded_physical_states, get_decoded_rewards
from evaluation.common.env_simulation import get_env_infos_from_actions, get_env_obs_from_phys_states
from evaluation.common.ood_phys_states import get_hardcoded_ood_state
from evaluation.common.rollouts import get_posteriors, get_priors, get_one_step_priors, get_most_id_state, get_random_posteriors_priors


def create_test_env(overrides_config):
    env_config = overrides_config.environment.env if "environment" in overrides_config else StateBasedDMCEnvFactory().get_default_config()
    test_env = StateBasedDMCEnvFactory().build(seed=overrides_config.seed,
                                               config=env_config)

    return test_env, env_config


def evaluate(experiment,
             overrides_config,
             warm_up_steps: int = 3,
             num_init_episodes: int = 10,
             rollout_length: int = 30,
             num_rollouts: int = 1000,
             start_state_path: str = None,
             analyze_id: bool = True,
             analyze_ood: bool = True,
             analyze_attr: bool = True,
             analyze_rew: bool = True,
             combined_phys_discr: bool = True,
             combined_rew_discr: bool = True,
             open_loop: bool = True):
    
    # Set seed for reproducibility.
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    test_env, env_config = create_test_env(overrides_config)
    
    mbrl_config = experiment._conf_save_dict["mbrl_config"]
    img_preprocessor = ImgPreprocessor(depth_bits=mbrl_config.img_preprocessing.color_depth_bits,
                                       add_cb_noise=mbrl_config.img_preprocessing.add_cb_noise)
    
    assert rollout_length <= 1000 / env_config.action_repeat, "[ERROR] Rollout length too long."
    assert warm_up_steps < rollout_length, "[ERROR] Too many warm-up steps."

    eval_data_collector = EvalDataCollector(env=test_env,
                                            policy=experiment._policy,
                                            model=experiment._model,
                                            sequences_per_collect=num_init_episodes,
                                            imagine_horizon=rollout_length)
    
    # Create directory for evaluation results.
    eval_dir = overrides_config.log_dir + "/eval"
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    
    # Define environment specific state mask.
    mask = test_env.get_translation_inv_mask(transformed=False)[None, None]
    
    # Load training dataset of physical states.
    dataset = torch.load(overrides_config.log_dir + "/dataset.pt")
    states = torch.cat([torch.stack(dataset[k]["states"], dim=0) for k in dataset.keys()], dim=0)
    
    args = (eval_data_collector, test_env, experiment, img_preprocessor, rollout_length, num_rollouts, warm_up_steps, open_loop, mask)
    
    ###########################################################################################
    # -------------------------------- ID --------------------------------
    if analyze_id or analyze_attr:
        print("[ID] Starting ID setting...")
        
        if start_state_path is not None:
            # Load state from somewhere else.
            print("[ID] Loading starting state from " + start_state_path + "/id_state.json")
            with open(start_state_path + "/id_state.json", "r") as f:
                id_state = json.load(f)
            most_id_state, most_id_mean_knn_dist = torch.tensor(id_state["state"]), id_state["mean_knn_dist"]
        else:
            # Get "most in-distribution" point from the training dataset of physical states.
            if not os.path.isfile(overrides_config.log_dir + "id_state.json"):
                print("[ID] ID state not yet found, searching for most ID state...")
                most_id_state, most_id_mean_knn_dist = get_most_id_state(dataset=states, frac=4, k=100)
                with open(overrides_config.log_dir + "id_state.json", "w") as out:
                    json.dump({"state": most_id_state.cpu().numpy().tolist(),
                               "mean_knn_dist": most_id_mean_knn_dist.item()},
                              out)
            else:
                print("[ID] Load already existing ID state from same directory.")
                with open(overrides_config.log_dir + "id_state.json", "r") as f:
                    id_state = json.load(f)
                most_id_state, most_id_mean_knn_dist = torch.tensor(id_state["state"]), id_state["mean_knn_dist"]
        
        dump_dir = eval_dir + "/id"
        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)
            
        id_post_infos, id_prior_infos = analyze_from_starting_state(*args,
                                                                    start_state=most_id_state,
                                                                    start_model_state=None,
                                                                    eval_dir=dump_dir)
        id_post_infos["start_phys_state"] = most_id_state
        id_prior_infos["start_phys_state"] = most_id_state
        
        torch.save(id_post_infos, overrides_config.log_dir + "id_post_infos.pt")
        torch.save(id_prior_infos, overrides_config.log_dir + "id_prior_infos.pt")
    else:
        id_post_infos, id_prior_infos = None, None
    
    # -------------------------------- OOD --------------------------------
    if analyze_ood:
        print("[OOD] Starting OOD setting...")
        
        ood_phys_state = get_hardcoded_ood_state(env_name=overrides_config.environment.env.env)
            
        if ood_phys_state is not None:
            ood_phys_state = test_env.transform_phys_state(ood_phys_state)
        
            dump_dir = eval_dir + "/ood"
            if not os.path.isdir(dump_dir):
                os.makedirs(dump_dir)
                
            ood_post_infos, ood_prior_infos  = analyze_from_starting_state(*args,
                                                                           start_state=ood_phys_state,
                                                                           start_model_state=None,
                                                                           eval_dir=dump_dir)
            
            ood_post_infos["start_phys_state"] = ood_phys_state
            ood_prior_infos["start_phys_state"] = ood_phys_state
            
            torch.save(ood_post_infos, overrides_config.log_dir + "ood_post_infos.pt")
            torch.save(ood_prior_infos, overrides_config.log_dir + "ood_prior_infos.pt")
        else:
            print("[OOD] Skip, since hardcoded OOD state not defined.")
            ood_post_infos, ood_prior_infos = None, None
    else:
        ood_post_infos, ood_prior_infos = None, None
    
    # -------------------------------- ATTRACTOR --------------------------------
    if analyze_attr:
        print("[ATTR] Starting attractor analysis...")
        
        dump_dir = eval_dir + "/attr"
        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)
        
        # Sequences to be plotted.
        id_seq = {k: v[0:1] for k, v in id_post_infos["states"].items()}
        ood_seq = {k: v[0:1] for k, v in ood_post_infos["states"].items()} if ood_post_infos is not None else None
        seqs = [id_seq, ood_seq] if ood_post_infos is not None else [id_seq]
        
        analyze_attractor(experiment=experiment,
                          data_collector=eval_data_collector,
                          sequences=seqs,
                          warm_up_steps=warm_up_steps,
                          rollout_length=rollout_length,
                          dir=dump_dir)
    
    # -------------------------------- REWARDS --------------------------------
    if analyze_rew:
        print("[REW] Starting rewards analysis...")
        
        dump_dir = eval_dir + "/rew"
        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)
            
        analyze_reward(experiment=experiment,
                       data_collector=eval_data_collector,
                       rollout_length=rollout_length,
                       warm_up_steps=warm_up_steps,
                       dir=dump_dir)
    
    # -------------------------------- COMBINED --------------------------------
    if combined_phys_discr:
        print("[COMB] Starting combined rewards analysis...")
        
        versions = ["id", "ood"] if not overrides_config.environment.env.env == "cartpole_swingup" else ["id"]
        for ver in versions:
            dump_dir = eval_dir + "/combined_" + ver
            if not os.path.isdir(dump_dir):
                os.makedirs(dump_dir)
                
            post_phys_states, prior_phys_states = [], []
            post_comp_phys_states, prior_comp_phys_states = [], []
            prior_uncertainties = []
            for seed in os.listdir(overrides_config.log_dir + "/../"):
                if not seed.isdigit():
                    continue
                
                # post
                infos = torch.load(overrides_config.log_dir + "/../" + str(seed) + "/" + ver + "_post_infos.pt")
                post_phys_states.append(infos["dec_phys_states"])
                post_comp_phys_states.append(infos["gt_phys_states"])
                
                # prior
                infos = torch.load(overrides_config.log_dir + "/../" + str(seed) + "/" + ver + "_prior_infos.pt")
                prior_phys_states.append(infos["dec_phys_states"])
                prior_comp_phys_states.append(infos["gt_phys_states"])
                prior_uncertainties.append(infos["unc"])
                
            phys_states = [torch.cat(post_phys_states, 0), torch.cat(prior_phys_states, 0)]
            comp_phys_states = [torch.cat(post_comp_phys_states, 0), torch.cat(prior_comp_phys_states, 0)]
            uncertainties = torch.cat(prior_uncertainties, 0)
            
            compute_phys_diff(env=test_env,
                              mask=mask,
                              warm_up_steps=warm_up_steps,
                              phys_states=phys_states,
                              comp_phys_states=comp_phys_states,
                              uncertainties=uncertainties,
                              diff_name="phys_diff_combined",
                              dir=dump_dir)
            
    if combined_rew_discr:
        print("[COMB] Starting combined reward analysis...")
        
        dump_dir = eval_dir + "/combined_rew"
        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)
            
        post_rewards, post_comp_rewards = [], []
        prior_rewards, prior_comp_rewards = [], []
        for seed in os.listdir(overrides_config.log_dir + "/../"):
            if not seed.isdigit():
                continue
            
            infos = torch.load(overrides_config.log_dir + "/../" + str(seed) + "/reward_infos.pt")
            
            # post
            post_rewards.append(infos["post_dec_reward"].cpu())
            post_comp_rewards.append(infos["post_gt_reward"].cpu())
            
            # prior
            prior_rewards.append(infos["prior_dec_reward"].cpu())
            prior_comp_rewards.append(infos["prior_gt_reward"].cpu())
            
        rewards = [torch.cat(post_rewards, 0), torch.cat(prior_rewards, 0)]
        comp_rewards = [torch.cat(post_comp_rewards, 0), torch.cat(prior_comp_rewards, 0)]
        
        compute_rew_diff(warm_up_steps=warm_up_steps,
                         rewards=rewards,
                         comp_rewards=comp_rewards,
                         name="reward_diff_combined",
                         dir=dump_dir)


def compute_phys_diff(env,
                      mask,
                      warm_up_steps,
                      gt_phys_states: Optional[torch.Tensor] = None,
                      phys_states: list[torch.Tensor] = [],
                      comp_phys_states: list[torch.Tensor] = [],
                      phys_states_labels: list[str] = ["Posterior", "Prior"],
                      uncertainties: Optional[torch.Tensor] = None,
                      colors: list[str] = ["limegreen", "royalblue", "darkorange", "forestgreen"],
                      diff_name: str = "phys_diff",
                      dev_name: str = "phys_state",
                      dir: str = "/."):
    
    # Untransform and mask all physical states.
    if gt_phys_states is not None:
        gt_phys_states = env.untransform_phys_state(gt_phys_states).cpu() * mask[0]
    for i in range(len(phys_states)):
        phys_states[i] = env.untransform_phys_state(phys_states[i]).cpu() * mask
        comp_phys_states[i] = env.untransform_phys_state(comp_phys_states[i]).cpu() * mask
    if uncertainties is not None:
        uncertainties = uncertainties.cpu()
    
    phys_dims = phys_states[0].shape[-1]
    
    # Only compute differences in positions.
    if gt_phys_states is not None:
        gt_phys_states = gt_phys_states[..., :int(phys_dims/2)]
    for i in range(len(phys_states)):
        phys_states[i] = phys_states[i][..., :int(phys_dims/2)]
        comp_phys_states[i] = comp_phys_states[i][..., :int(phys_dims/2)]
    
    # Compute dim-wise difference to accomodate wrapping for angles < -pi and > pi.
    is_angle = env.get_is_angle()    
    diffs = []
    for i in range(len(phys_states)):
        diff = []
        for dim in range(phys_states[0].shape[-1]):
            if dim < len(is_angle) and is_angle[dim]:
                dim_diff = abs(np.mod(phys_states[i][..., dim] - comp_phys_states[i][..., dim] + np.pi, 2*np.pi) - np.pi) # difference in rad
            else:
                dim_diff = abs(phys_states[i][..., dim] - comp_phys_states[i][..., dim])
            diff.append(dim_diff)
        # diff = [dims, num_seq, seq_len]
        dists = torch.sqrt(torch.tensor(np.array(diff)).mean(0))
        diffs.append(dists) 
    
    diff_means = [d.mean(0) for d in diffs]
    diff_stds = [d.std(0) for d in diffs]
    if uncertainties is not None:
        diff_means += [uncertainties.squeeze(-1).mean(0)]
        diff_stds += [uncertainties.squeeze(-1).std(0)]
    
    # Plot total mean physical state discrepancies.
    labels = [phys_states_labels[i] + " Phys. State Diff." for i in range(len(phys_states))]
    labels += ["Uncertainty"]
    plot_funcs_shared(means=diff_means,
                      stds=diff_stds,
                      labels=labels,
                      colors=colors[1:],
                      linestyles=["-"] * (len(phys_states) + 1),
                      linewidths=[2] * (len(phys_states) + 1),
                      vline=warm_up_steps-1,
                      y_lim=[0, 3],
                      y_lim2=[0, 10] if uncertainties is not None else None,
                      y_axis_2=2 if uncertainties is not None else None,
                      y_label_1="Phys. Discrepancy",
                      y_label_2="Prior Uncertainty" if uncertainties is not None else None,
                      title="Development of Masked Phys. State Difference",
                      path=dir + "/" + diff_name)        
    
    # Plot total mean physical state development.
    if gt_phys_states is not None:
        means = [gt_phys_states.mean(-1).mean(0)] + [ps.mean(-1).mean(0) for ps in phys_states]
        stds = [gt_phys_states.mean(-1).std(0)] + [ps.mean(-1).std(0) for ps in phys_states]
        labels = ["GT Phys. State"] + [phys_states_labels[i] + " Phys. State" for i in range(len(phys_states))]
        plot_funcs_shared(means=means,
                          stds=stds,
                          labels=["GT Phys. State", "Posterior Phys. State", "Prior Phys. State"],
                          colors=colors[:-1],
                          linestyles=["--"] + ["-"] * len(phys_states),
                          linewidths=[5, 2, 2],
                          vline=warm_up_steps-1,
                          title="Development of Masked Phys. State",
                          path=dir + "/" + dev_name)
    
    # Save physical differences to csv.
    if uncertainties is not None:
        header_names = ["mean_post", "mean_prior", "mean_prior_unc"] + ["std_post", "std_prior", "std_prior_unc"]
    else:
        header_names = ["mean_post", "mean_prior"] + ["std_post", "std_prior"]
    diff_stds
    save_to_csv(data=diff_means + diff_stds,
                header_names=header_names,
                path=dir,
                file_name=diff_name + ".csv")
    
    return diffs


def compute_rew_diff(warm_up_steps,
                     rewards: list[torch.Tensor] = [],
                     comp_rewards: list[torch.Tensor] = [],
                     rewards_labels: list[str] = ["Posterior", "Prior"],
                     colors: list[str] = ["royalblue", "darkorange"],
                     name: str = "reward_diff",
                     dir: str = "/."):
    
    diffs = [(rewards[i] - comp_rewards[i]).squeeze(-1).cpu() for i in range(len(rewards))]
    diff_means = [d.mean(0) for d in diffs]
    diff_stds = [d.std(0) for d in diffs]
    
    # Plot total mean physical state discrepancies.
    labels = [rewards_labels[i] + " Reward Diff." for i in range(len(rewards))]
    plot_funcs_shared(means=diff_means,
                      stds=diff_stds,
                      labels=labels,
                      colors=colors,
                      linestyles=["-"] * len(rewards),
                      linewidths=[2] * len(rewards),
                      vline=warm_up_steps-1,
                      y_lim=[-1.5, 1.5],
                      title="Development of Reward Difference",
                      path=dir + "/" + name)     
    
    # Dump differences to csv.
    save_to_csv(data=diff_means+diff_stds,
                header_names=["mean_post", "mean_prior", "std_post", "std_prior"],
                path=dir,
                file_name=name + ".csv")   
    
    return diffs


def dump_obs_and_videos(data_collector, 
                        img_preprocessor,
                        experiment,
                        gt_obs_seq,
                        prior_state_seq,
                        prior_recon_phys_seq,
                        prior_act_phys_seq,
                        post_state_seq,
                        post_phys_seq,
                        dir="/."):
    
    # ............ GROUND-TRUTH ............
    # Dump gt observation sequence.
    preprocessed_obs = img_preprocessor(gt_obs_seq)
    save_obs_trajectory(preprocessed_obs,
                        path=dir + "/gt_obs/",
                        img_preprocessor=img_preprocessor)
    
    # ............ PRIOR ............
    # Dump prior reconstructed observation sequence.
    recon_prior_obs = get_reconstructions(experiment, prior_state_seq)[0]
    save_obs_trajectory(recon_prior_obs.squeeze(0),
                        path=dir + "/prior_recon_obs/",
                        img_preprocessor=img_preprocessor)
              
    # Dump observations as simulated in environment given prior physical states.
    env_prior_obs = get_env_obs_from_phys_states(data_collector=data_collector,
                                                 phys_states=prior_recon_phys_seq,
                                                 normed=False)
    preproc_env_prior_obs = img_preprocessor(env_prior_obs)
    save_obs_trajectory(preproc_env_prior_obs.squeeze(0),
                        path=dir + "/prior_recon_phys_obs/",
                        img_preprocessor=img_preprocessor)
    
    # Dump observations as simulated in environment from prior actions
    env_act_prior_obs = get_env_obs_from_phys_states(data_collector=data_collector,
                                                     phys_states=prior_act_phys_seq,
                                                     normed=False)
    preproc_env_prior_obs = img_preprocessor(env_act_prior_obs)
    save_obs_trajectory(preproc_env_prior_obs.squeeze(0),
                        path=dir + "/prior_act_phys_obs/",
                        img_preprocessor=img_preprocessor)
    
    # ............ POSTERIOR ............
    # Dump posterior reconstructed observation sequence.
    recon_prior_obs = get_reconstructions(experiment, post_state_seq)[0]
    save_obs_trajectory(recon_prior_obs.squeeze(0),
                        path=dir + "/post_recon_obs/",
                        img_preprocessor=img_preprocessor)
    
    # Dump observations as simulated in environment given posterior physical states
    env_post_obs = get_env_obs_from_phys_states(data_collector=data_collector,
                                                phys_states=post_phys_seq,
                                                normed=False)
    preproc_env_post_obs = img_preprocessor(env_post_obs)
    save_obs_trajectory(preproc_env_post_obs.squeeze(0),
                        path=dir + "/post_recon_phys_obs/",
                        img_preprocessor=img_preprocessor)


def get_random_rollouts(data_collector,
                        warm_up_steps,
                        rollout_length=100,
                        num_searches=1000,
                        dir="/."):
    """
    Either collect a set of random posterior and prior rollouts or load already collected set.
    """
    
    
    path_to_rollouts = dir + "/../../random_model_rollouts_100.pt"
    if not os.path.isfile(path_to_rollouts):
        random_model_rollouts = get_random_posteriors_priors(data_collector,
                                                             rollout_length=rollout_length+warm_up_steps,
                                                             num_searches=num_searches)
        torch.save(random_model_rollouts, path_to_rollouts)
    else:
        random_model_rollouts = torch.load(path_to_rollouts)
    
    return random_model_rollouts


def analyze_from_starting_state(data_collector,
                                env,
                                experiment,
                                img_preprocessor,
                                rollout_length,
                                num_rollouts,
                                warm_up_steps,
                                open_loop,
                                mask,
                                start_state,
                                start_model_state,
                                eval_dir):
    
    # ............ POSTERIOR ............
    # Posterior rollouts have sequence length rollout_length + warm_up_steps for a fair comparison to prior rollouts.
    # Compute one "gt" posterior rollout.
    start_phys_state = start_state
    gt_post_states, gt_post_phys_states, gt_post_obs, gt_post_act, gt_post_rew = get_posteriors(data_collector=data_collector,
                                                                                                start_model_state=start_model_state,
                                                                                                start_phys_state=start_phys_state,
                                                                                                num_rollouts=num_rollouts,
                                                                                                rollout_length=rollout_length + warm_up_steps + 1,
                                                                                                pred_actions=None,
                                                                                                sample_model=True)
    
    # Compensate offset between environment info (obs, physical states) and model outputs (posterior states, actions).
    gt_post_states = {k: v[:, 1:] for k, v in gt_post_states.items()}
    gt_post_phys_states = gt_post_phys_states[:, :-1]
    gt_post_obs = [o[:, :-1] for o in gt_post_obs]
    gt_post_act = gt_post_act[:, 1:]
    
    # Compute one one-step priors (with posterior background), which is the state before updated with the current observation.
    post_states = get_one_step_priors(eval_data_collector=data_collector,
                                      posterior_states=gt_post_states,
                                      gt_actions=gt_post_act,
                                      rollout_length=rollout_length + warm_up_steps,
                                      num_rollouts=1,
                                      open_loop=open_loop)
    
    # Get "baseline" decoder physical states from posterior rollouts.
    post_phys_states = get_decoded_physical_states(experiment, post_states)
    
    # Get comparison physical states for posterior ones.
    post_comp_phys_states = gt_post_phys_states.to(post_phys_states.device)
    
    # ............ PRIOR ............
    # Start prior rollouts with a fixed amount of warm-up steps to properly initialize model state before imagination.
    start_model_state = {k: v[0, warm_up_steps+1].unsqueeze(0) for k, v in gt_post_states.items()}
    prior_states, prior_act = get_priors(data_collector=data_collector,
                                         pred_actions=gt_post_act[0, warm_up_steps:].unsqueeze(0) if open_loop else None,
                                         start_model_state=start_model_state,
                                         num_rollouts=num_rollouts,
                                         sample_model=True)         # act upon sampled model states

    # Pad first steps with posterior samples for easier visualization.
    post_pad = {k: v[:, :warm_up_steps] for k, v in post_states.items()}
    prior_states = {k: torch.cat((post_pad[k], v), 1) for k, v in prior_states.items()}
    gt_post_act_pad = gt_post_act[:, :warm_up_steps].to(prior_act.device)
    prior_act = torch.cat((gt_post_act_pad, prior_act), 1)
    
    # Get "predicted" decoder physical states from prior rollouts.
    prior_phys_states = get_decoded_physical_states(experiment, prior_states)
        
    # Get prior gt physical states and rewards to compare to by simulating the environment with given prior actions.
    _, _, prior_comp_phys_states = get_env_infos_from_actions(data_collector=data_collector,
                                                              actions=prior_act,
                                                              start_phys_state=start_phys_state,
                                                              phys_state_is_obs=False)
    prior_comp_phys_states = prior_comp_phys_states.to(prior_phys_states.device)
    
    # ............ UNCERTAINTY ............
    # Compute prior uncertainties, if applicable.
    if experiment._ensemble is not None:
        prior_uncertainties = get_uncertainties(experiment=experiment,
                                                states=prior_states,
                                                actions=prior_act,
                                                rollout_length=rollout_length)
    else:
        prior_uncertainties = None
    
    # ............ PLOTS ............
    # Plot discrepancies.
    _ = compute_phys_diff(env=env,
                          mask=mask,
                          warm_up_steps=warm_up_steps,
                          gt_phys_states=gt_post_phys_states.squeeze(0),
                          phys_states=[post_phys_states, prior_phys_states],
                          comp_phys_states=[post_comp_phys_states, prior_comp_phys_states],
                          uncertainties=prior_uncertainties,
                          dir=eval_dir)
    
    # Render observations and videos.
    # priors
    idx = 0
    prior_states_seq_max = {k: v[idx].unsqueeze(0) for k, v in prior_states.items()}
    prior_phys_seq_max = env.untransform_phys_state(prior_phys_states[idx]).cpu() * mask
    prior_act_phys_seq_max = env.untransform_phys_state(prior_comp_phys_states[idx]).cpu() * mask
    # post
    post_states_seq = {k: v[idx].unsqueeze(0) for k, v in prior_states.items()}
    post_phys_seq = env.untransform_phys_state(post_phys_states[idx]).cpu() * mask
    dump_obs_and_videos(data_collector=data_collector, 
                        img_preprocessor=img_preprocessor,
                        experiment=experiment,
                        gt_obs_seq=gt_post_obs[0][idx],
                        prior_state_seq=prior_states_seq_max,
                        prior_recon_phys_seq=prior_phys_seq_max,
                        prior_act_phys_seq=prior_act_phys_seq_max,
                        post_state_seq=post_states_seq,
                        post_phys_seq=post_phys_seq,
                        dir=eval_dir)
    
    post_infos = {
        "states": post_states,
        "act": gt_post_act,
        "dec_phys_states": post_phys_states,
        "gt_phys_states": post_comp_phys_states
    }
    prior_infos = {
        "states": prior_states,
        "act": prior_act,
        "dec_phys_states": prior_phys_states,
        "gt_phys_states": prior_comp_phys_states,
        "unc": prior_uncertainties
    }
    
    return post_infos, prior_infos


def analyze_attractor(experiment,
                      data_collector,
                      sequences,
                      warm_up_steps,
                      rollout_length=None,
                      dir="/."):
    
    random_model_rollouts = get_random_rollouts(data_collector=data_collector,
                                                warm_up_steps=warm_up_steps,
                                                dir=dir)
    
    post_states, prior_states = random_model_rollouts["post_states"], random_model_rollouts["prior_states"]
    
    # Use specified rollout length, if defined.
    if rollout_length is not None:
        post_states = {k: v[:, :rollout_length+warm_up_steps] for k, v in post_states.items()}
        prior_states = {k: v[:, :rollout_length+warm_up_steps] for k, v in prior_states.items()}
        sequences = [{k: v[:, :rollout_length+warm_up_steps] for k, v in seq.items()} for seq in sequences]
    
    num_seq, seq_len = experiment._model.get_deterministic_features(post_states).shape[:2]
    
    # Define a state to be the deterministic sample.
    det_seqs = [experiment._model.get_deterministic_features(seq).cpu() for seq in sequences]
    det_prior_states = experiment._model.get_deterministic_features(prior_states).cpu()
    det_post_states = experiment._model.get_deterministic_features(post_states).cpu()
    
    # Normalize.
    ref_states = torch.cat((det_post_states, det_prior_states), dim=0)
    max_per_dim = torch.max(torch.max(ref_states, dim=0, keepdim=True).values, dim=1, keepdim=True).values
    min_per_dim = torch.min(torch.min(ref_states, dim=0, keepdim=True).values, dim=1, keepdim=True).values
    
    denom = max_per_dim - min_per_dim
    valid = (denom > 1e-8).squeeze(0).squeeze(0)
    ref_states = ((ref_states - min_per_dim) / denom)[..., valid]
    det_prior_states = ((det_prior_states - min_per_dim) / denom)[..., valid]
    det_post_states = ((det_post_states - min_per_dim) / denom)[..., valid]
    det_seqs = [((seq - min_per_dim) / denom)[..., valid] for seq in det_seqs]
    
    # Determine dims after valid extraction.
    dims = valid.sum()
    
    # Perfom PCA.
    proj = PCA(n_components=2)
    proj = proj.fit(ref_states.reshape(-1, dims))
    
    # Transform.
    transform_seqs = [proj.transform(seq.reshape(-1, dims)).reshape(1, seq_len, 2) for seq in det_seqs]
    transform_prior_states = proj.transform(det_prior_states.reshape(-1, dims)).reshape(num_seq, seq_len, 2)
    transform_post_states = proj.transform(det_post_states.reshape(-1, dims)).reshape(num_seq, seq_len, 2)
    
    # Plot.
    states_to_plot = torch.cat((torch.from_numpy(transform_prior_states), torch.from_numpy(transform_post_states)), 0)
    
    # Normalize between -1 and 1 for better visualization.
    max_per_dim = torch.max(torch.max(states_to_plot, dim=0, keepdim=True).values, dim=1, keepdim=True).values
    min_per_dim = torch.min(torch.min(states_to_plot, dim=0, keepdim=True).values, dim=1, keepdim=True).values
    
    denom = max_per_dim - min_per_dim
    states_to_plot = 2 * (states_to_plot - min_per_dim) / denom - 1
    seqs_to_plot = [2 * (torch.from_numpy(seq) - min_per_dim) / denom - 1 for seq in transform_seqs]
    
    # Plot all lines + (same) vector field in separate subplots
    plot_vector_field_with_lines(pts=states_to_plot,
                                 lines=[seq[0] for seq in seqs_to_plot],
                                 lines_cmaps=["viridis", "plasma"],
                                 n_bins=20,
                                 min_bin_count=1,
                                 num_plots=2,
                                 path=dir,
                                 colorbar=True,
                                 figname="pca_plot")
    
    
def analyze_reward(experiment,
                   data_collector,
                   rollout_length,
                   warm_up_steps,
                   dir="/."):
    
    random_model_rollouts = get_random_rollouts(data_collector=data_collector,
                                                warm_up_steps=warm_up_steps,
                                                dir=dir)
        
    post_states, post_act, post_start_phys_state = random_model_rollouts["post_states"], random_model_rollouts["post_act"], random_model_rollouts["post_start_phys_state"]
    prior_states, prior_act, prior_start_phys_state = random_model_rollouts["prior_states"], random_model_rollouts["prior_act"], random_model_rollouts["prior_start_phys_state"]
    
    assert rollout_length+warm_up_steps <= post_states["gru_cell_state"].shape[1]
    
    post_states = {k: v[:, :rollout_length+warm_up_steps] for k, v in post_states.items()}
    post_act = post_act[:, :rollout_length+warm_up_steps]
    
    prior_states = {k: v[:, :rollout_length+warm_up_steps] for k, v in prior_states.items()}
    prior_act = prior_act[:, :rollout_length+warm_up_steps]
    
    # ............ POSTERIOR ............
    post_decoded_rew = get_decoded_rewards(experiment, post_states)
    _, post_gt_rew, _ = get_env_infos_from_actions(data_collector=data_collector,
                                                   actions=post_act,
                                                   start_phys_state=post_start_phys_state,
                                                   phys_state_is_obs=False)
    
    # ............ PRIOR ............
    prior_decoded_rew = get_decoded_rewards(experiment, prior_states)
    _, prior_gt_rew, _ = get_env_infos_from_actions(data_collector=data_collector,
                                                    actions=prior_act,
                                                    start_phys_state=prior_start_phys_state,
                                                    phys_state_is_obs=False)
    
    reward_infos = {
        "post_dec_reward": post_decoded_rew,
        "post_gt_reward": post_gt_rew,
        "prior_dec_reward": prior_decoded_rew,
        "prior_gt_reward": prior_gt_rew
    }
    torch.save(reward_infos, dir + "/../../reward_infos.pt")
    
    compute_rew_diff(warm_up_steps=warm_up_steps,
                     rewards=[post_decoded_rew.cpu(), prior_decoded_rew.cpu()],
                     comp_rewards=[post_gt_rew.cpu(), prior_gt_rew.cpu()],
                     dir=dir)