
import os
from typing import cast
import argparse
import pathlib
import hydra.utils
import numpy as np
import torch
import random

import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.env
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    per_step_loss_threshold: np.ndarray,
    max_loss_threshold: np.ndarray,
):
    with torch.no_grad():
        num_added = 0
        sampling_round = 0
        transitions = replay_buffer.sample(batch_size=batch_size).astuple()
        obs_dataset = transitions[0]
        quantization_bits = np.log2(1e-6)

        total_initial_states = 0
        complete_rollouts = []
        while num_added < batch_size:
            batch_size_ = batch_size - int(num_added)
            total_initial_states += batch_size_

            if sampling_round > 0:
                batch = replay_buffer.sample(batch_size_)
                initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
                obs = initial_obs
            else:
                obs = obs_dataset
            obs_dims = obs.shape[-1]

            lost_info = np.zeros((batch_size_, obs_dims + 1))

            model_state = {}
            rollout_tracker = np.zeros(batch_size_)
            complete_rollouts.append(rollout_tracker)
            curr_idx = np.arange(batch_size_)
            for step in range(rollout_horizon):
                accum_dones = np.zeros(obs.shape[0], dtype=bool)
                action = agent.act(obs, sample=sac_samples_action, batched=True)
                model_state["obs"] = obs
                pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
                    action, model_state, sample=True
                )
                truncateds = np.zeros_like(pred_dones, dtype=bool)
                conditional_var = next_model_state["conditional_var"].cpu().numpy()

                lost_info_this_step = np.clip(
                    0.5 * np.log2(2 * np.pi * np.exp(1) * conditional_var) - quantization_bits, 0, np.inf
                )
                lost_info += lost_info_this_step

                accum_dones |= (lost_info_this_step > per_step_loss_threshold).any(axis=-1)
                accum_dones |= (lost_info > max_loss_threshold).any(axis=-1)

                sac_buffer.add_batch(
                    obs[~accum_dones],
                    action[~accum_dones],
                    pred_next_obs[~accum_dones],
                    pred_rewards[~accum_dones, 0],
                    pred_dones[~accum_dones, 0],
                    truncateds[~accum_dones, 0],
                    lost_info_this_step[~accum_dones],
                    lost_info[~accum_dones],
                    np.ones_like(lost_info[~accum_dones]),
                )
                num_added += (~accum_dones).sum()

                rollout_tracker[curr_idx[~accum_dones]] += 1

                accum_dones |= pred_dones.squeeze()
                obs = pred_next_obs[~accum_dones]
                if len(obs) == 0:
                    break
                lost_info = lost_info[~accum_dones]
                curr_idx = curr_idx[~accum_dones]

            sampling_round += 1
            if sampling_round == 1:
                assert num_added > 0
        rollout_tracker = np.concatenate(complete_rollouts)
        sac_buffer.re_compute_sampling_idxs()
        return num_added


def rollout_model_and_populate_sac_buffer_no_trunc(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    per_step_loss_threshold: np.ndarray,
    max_loss_threshold: np.ndarray,
):
    # collect all observations and uncertainties over time
    all_obs = []
    all_acts = []
    with torch.no_grad():
        num_added = 0
        sampling_round = 0
        transitions = replay_buffer.sample(batch_size=batch_size).astuple()
        obs_dataset = transitions[0]
        quantization_bits = np.log2(1e-6)

        total_initial_states = 0
        complete_rollouts = []
        while num_added < batch_size:
            batch_size_ = batch_size - int(num_added)
            total_initial_states += batch_size_

            if sampling_round > 0:
                batch = replay_buffer.sample(batch_size_)
                initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
                obs = initial_obs
            else:
                obs = obs_dataset
            obs_dims = obs.shape[-1]

            lost_info = np.zeros((batch_size_, obs_dims + 1))
            
            uncertainties_over_time = np.zeros((obs.shape[0], rollout_horizon))

            model_state = {}
            rollout_tracker = np.zeros(batch_size_)
            complete_rollouts.append(rollout_tracker)
            curr_idx = np.arange(batch_size_)
            for step in range(rollout_horizon):
                accum_dones = np.zeros(obs.shape[0], dtype=bool)
                action = agent.act(obs, sample=sac_samples_action, batched=True)
                model_state["obs"] = obs
                pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
                    action, model_state, sample=True
                )
                truncateds = np.zeros_like(pred_dones, dtype=bool)
                conditional_var = next_model_state["conditional_var"].cpu().numpy()

                lost_info_this_step = np.clip(
                    0.5 * np.log2(2 * np.pi * np.exp(1) * conditional_var) - quantization_bits, 0, np.inf
                )
                lost_info += lost_info_this_step
                
                accum_dones = np.zeros((lost_info_this_step.shape[0], ), dtype=bool)
                
                uncertainties_over_time[:, step] = lost_info_this_step.mean(-1)
                all_obs.append(obs)
                all_acts.append(action)

                sac_buffer.add_batch(
                    obs[~accum_dones],
                    action[~accum_dones],
                    pred_next_obs[~accum_dones],
                    pred_rewards[~accum_dones, 0],
                    pred_dones[~accum_dones, 0],
                    truncateds[~accum_dones, 0],
                    lost_info_this_step[~accum_dones],
                    lost_info[~accum_dones],
                    np.ones_like(lost_info[~accum_dones]),
                )
                num_added += (~accum_dones).sum()

                rollout_tracker[curr_idx[~accum_dones]] += 1

                accum_dones |= pred_dones.squeeze()
                obs = pred_next_obs[~accum_dones]
                if len(obs) == 0:
                    break
                lost_info = lost_info[~accum_dones]
                curr_idx = curr_idx[~accum_dones]

            sampling_round += 1
            if sampling_round == 1:
                assert num_added > 0
        rollout_tracker = np.concatenate(complete_rollouts)
        sac_buffer.re_compute_sampling_idxs()
        
        # Find most uncertain rollout.
        all_obs = np.stack(all_obs, 1)
        all_acts = np.stack(all_acts, 1)
        
        max_unc_idx = np.argmax(uncertainties_over_time.mean(-1), 0)
        min_unc_idx = np.argmin(uncertainties_over_time.mean(-1), 0)
        
        most_uncertain_rollout = all_obs[max_unc_idx]
        most_certain_rollout = all_obs[min_unc_idx]
        
        most_uncertain_rollout_acts = all_acts[max_unc_idx]
        most_certain_rollout_acts = all_acts[min_unc_idx]
        
        most_uncertain_rollout_uncertainty = uncertainties_over_time[max_unc_idx]
        most_certain_rollout_uncertainty = uncertainties_over_time[min_unc_idx]
        
        return num_added, most_uncertain_rollout, most_uncertain_rollout_acts, most_uncertain_rollout_uncertainty, most_certain_rollout, most_certain_rollout_acts, most_certain_rollout_uncertainty


def collect_model_rollouts(args):
    load_path = args.load_path
    sac_buffer_capacity = args.buffer_capacity

    # ------------------- Load config -------------------
    cfg = mbrl.util.common.load_hydra_cfg(load_path)

    # ------------------- Create environment -------------------
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # ------------------- Load dynamics model -------------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg, obs_shape, act_shape, model_dir=load_path
    )

    # ------------------- Load agent -------------------
    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, 
             hydra.utils.instantiate(cfg.algorithm.agent, action_space=env.action_space))
    )
    agent.sac_agent.load_checkpoint(
        ckpt_path=os.path.join(load_path, "sac.pth"), evaluate=True
    )

    # ------------------- Load replay buffer -------------------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    if "100k" in load_path[-4:]:
        cfg.overrides.num_steps = 105000 # for halfcheetah_100k
    elif "300k" in load_path[-4:]:
        cfg.overrides.num_steps = 305000 # for halfcheetah_300k
    else:
        pass # should be fine for halfcheetah_10k
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
        load_dir=load_path,
    )

    # ------------------- Load thresholds -------------------
    thresholds = np.load(os.path.join(load_path, "thresholds.npz"))
    per_step_loss_threshold = thresholds["per_step_loss_threshold"]
    max_loss_threshold = thresholds["max_loss_threshold"]

    print(f"Loaded dynamics model, agent, replay buffer ({replay_buffer.num_stored} transitions), "
        f"and thresholds from {load_path}")
    print(f"per_step_loss_threshold shape: {per_step_loss_threshold.shape}")
    print(f"max_loss_threshold shape:      {max_loss_threshold.shape}")

    # ------------------- Create model environment -------------------
    torch_generator = torch.Generator(device=cfg.device)
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, term_fn, None, generator=torch_generator
    )

    # ------------------- Create SAC buffer -------------------
    # Use the final rollout_schedule value to get end-of-training rollout length
    num_epochs = cfg.overrides.num_steps // cfg.overrides.epoch_length
    rollout_length = int(
        mbrl.util.math.truncated_linear(
            *(cfg.overrides.rollout_schedule + [num_epochs])
        )
    )

    rng = np.random.default_rng(seed=cfg.seed)
    sac_buffer = mbrl.util.InfoReplayBuffer(sac_buffer_capacity, obs_shape, act_shape, rng=rng)

    print(f"Created model_env and sac_buffer (capacity={sac_buffer_capacity}, rollout_length={rollout_length})")

    # ------------------- Fill SAC buffer with model rollouts -------------------
    print("Fill SAC buffer with model rollouts...")
    while sac_buffer.num_stored < sac_buffer_capacity:
        num_added = rollout_model_and_populate_sac_buffer(
            model_env,
            replay_buffer,
            agent,
            sac_buffer,
            sac_samples_action=cfg.algorithm.sac_samples_action,
            rollout_horizon=rollout_length,
            batch_size=2048,
            per_step_loss_threshold=per_step_loss_threshold,
            max_loss_threshold=max_loss_threshold,
        )
        print(f"SAC buffer filled: {num_added} transitions added ({sac_buffer.num_stored} stored)")

    save_dir = os.path.join(load_path, "sac_buffer_dump")
    os.makedirs(save_dir, exist_ok=True)

    sac_buffer.save(save_dir)
    print(f"SAC buffer saved to {save_dir}")
            
    # ------------------- In- and Out-of-distribution model rollout -------------------
    print("Determine most certain and uncertain trajectories...")
    rng = np.random.default_rng(seed=cfg.seed)
    dummy_sac_buffer = mbrl.util.InfoReplayBuffer(sac_buffer_capacity, obs_shape, act_shape, rng=rng)
    
    most_ood_trajectory, most_ood_trajectory_acts = None, None
    most_id_trajectory, most_id_trajectory_acts = None, None
    highest_uncertainty, lowest_uncertainty = np.array([-float("inf")]), np.array([float("inf")])
    for _ in range(20):
        num_added, most_uncertain_rollout, most_uncertain_rollout_acts, most_uncertain_rollout_uncertainty, most_certain_rollout, most_certain_rollout_acts, most_certain_rollout_uncertainty = rollout_model_and_populate_sac_buffer_no_trunc(
            model_env,
            replay_buffer,
            agent,
            dummy_sac_buffer,
            sac_samples_action=cfg.algorithm.sac_samples_action,
            rollout_horizon=50,
            batch_size=2048,
            per_step_loss_threshold=per_step_loss_threshold,
            max_loss_threshold=max_loss_threshold,
        )
        print(f"SAC buffer filled: {num_added} transitions added ({sac_buffer.num_stored} stored)")
        
        if highest_uncertainty.sum() < most_uncertain_rollout_uncertainty.sum():
            most_ood_trajectory = most_uncertain_rollout
            most_ood_trajectory_acts = most_uncertain_rollout_acts
            highest_uncertainty = most_uncertain_rollout_uncertainty
        
        if lowest_uncertainty.sum() > most_certain_rollout_uncertainty.sum():
            most_id_trajectory = most_certain_rollout
            most_id_trajectory_acts = most_certain_rollout_acts
            lowest_uncertainty = most_certain_rollout_uncertainty
            
    save_dir = os.path.join(load_path, "most_uncertain_trajectory")
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez(pathlib.Path(save_dir) / "infos.npz",
             obs_trajectory=most_ood_trajectory,
             act_trajectory=most_ood_trajectory_acts,
             uncertainties=highest_uncertainty)
    
    save_dir = os.path.join(load_path, "most_certain_trajectory")
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez(pathlib.Path(save_dir) / "infos.npz",
             obs_trajectory=most_id_trajectory,
             act_trajectory=most_id_trajectory_acts,
             uncertainties=lowest_uncertainty)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", default="exp/halfcheetah_300k")
    parser.add_argument("--buffer_capacity", default=500000, type=int)
    args = parser.parse_args()
    
    collect_model_rollouts(args)