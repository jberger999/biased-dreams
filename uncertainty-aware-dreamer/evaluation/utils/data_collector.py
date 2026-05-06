import torch
from typing import Union

from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.data_collector import DataCollector
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy
from uncertainty_aware_dreamer.ssm_mbrl.util.stack_util import stack_maybe_nested_dicts


class EvalDataCollector(DataCollector):

    def __init__(self,
                 env,
                 policy: AbstractPolicy,
                 model,
                 sequences_per_collect: int,
                 imagine_horizon: int = 15,
                 obs_idx: int = 0):
        
        super().__init__(env=env,
                         policy=policy,
                         sequences_per_collect=sequences_per_collect,
                         max_sequence_length=-1,
                         action_noise_std=0.0) # we do not use action_noise_std
        self._imagine_horizon = imagine_horizon
        self._model = model
        self._obs_idx = obs_idx

    def collect(self, sample_policy=False, sample_model=False):
        """
        Collect data, where we incorporate the observations to update the model state.
        We always start from the start state provided by the environment.
        """
        with torch.no_grad():
            all_observations, all_actions, all_rewards, all_infos, all_post_states = [], [], [], [], []
            for i in range(self._sequences_per_collect):
                observations, actions, rewards, infos, post_states = self._rollout_policy(sample_policy=sample_policy,
                                                                                          sample_model=sample_model)
                all_observations.append(observations)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_infos.append(infos)
                all_post_states.append(post_states)

            return all_observations, all_actions, all_rewards, all_infos, all_post_states
        
    def rollout_policy(self, max_length=-1, start_phys_state=None, start_model_state=None, start_action=None, sample_policy=False, sample_model=False):
        """
        Rollout the policy in the environment for one episode or until the maximum sequence length is reached.
        """
        with torch.no_grad():
            observations, actions, rewards, infos, post_states = self._rollout_policy(max_length=max_length,
                                                                                        start_phys_state=start_phys_state,
                                                                                        start_model_state=start_model_state,
                                                                                        start_action=start_action,
                                                                                        sample_policy=sample_policy,
                                                                                        sample_model=sample_model)
            return observations, actions, rewards, infos, post_states
        
    def _rollout_policy(self, max_length=-1, start_phys_state=None, start_model_state=None, start_action=None, sample_policy=False, sample_model=False):
        """
        Rollout the policy in the environment for one episode or until the maximum sequence length is reached.
        """
        observations, actions, rewards, infos, post_states = [], [], [], [], []

        # Reset environment to start physical state, if available.
        obs, info = (
            self._env.reset()
            if start_phys_state is None
            else self._env.reset_to_state(start_phys_state)
        )
        # Use given start model state, if available.
        policy_state = (
            self._policy.get_initial(batch_size=1)[1]
            if start_model_state is None
            else start_model_state
        )
        # Use given start action, if available.
        action = (
            self._policy.get_initial(batch_size=1)[0]
            if start_action is None
            else start_action
        )

        observations.append(obs)
        actions.append(torch.squeeze(action, dim=0).cpu())
        rewards.append(torch.FloatTensor([0.0]))
        infos.append(info)
        post_states.append(policy_state)

        done = False
        i = 0

        while not done and (max_length < 0 or i < max_length-1):
            obs_for_pol = self._prepare_observation_for_policy(observation=obs)
            action_for_pol = self._prepare_action_for_policy(action=action)
            if sample_model:
                # Act upon sampled model state
                action_from_pol, policy_state = self._policy(observation=obs_for_pol,
                                                             prev_action=action_for_pol,
                                                             policy_state=policy_state,
                                                             sample=sample_policy) # return the mean action, if sample is False
            else:
                # Act upon deterministic model state
                action_from_pol, policy_state = self._policy.act_on_deterministic_state(observation=obs_for_pol,
                                                                                        prev_action=action_for_pol,
                                                                                        policy_state=policy_state,
                                                                                        sample=sample_policy) # return the mean action, if sample is False
            action = torch.squeeze(action_from_pol, dim=0).cpu()

            obs, reward, done, info = self._env.step(action=action)

            i += 1
            reward = torch.FloatTensor([reward])

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            post_states.append(policy_state)
   
        observations = [torch.stack([o[i] for o in observations], dim=0) for i in range(len(observations[0]))]

        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        infos = stack_maybe_nested_dicts(infos, dim=0)
        post_states = stack_maybe_nested_dicts(post_states, dim=0)
        post_states = {k: v.squeeze(1) for k, v in post_states.items()}

        return observations, actions, rewards, infos, post_states

    def imagine_rollout(self, start_model_states, sample_policy=False, sample_model=False, include_first=True):
        """
        Rollout policy on priors only, which corresponds to no intergration of the environment observation into the model states.
        """
        policy_fn = self._policy.actor.get_sampled_action_and_log_prob if sample_policy else self._policy.actor.get_greedy_action_and_log_prob
        with torch.no_grad():
            # Rollout policy for a certain amount of steps without incorporating observations.
            if sample_model:
                # Act upon sampled model states
                imagined_states, action_log_probs, actions = self._model.rollout_policy(state=start_model_states,
                                                                                        policy_fn=policy_fn,
                                                                                        num_steps=self._imagine_horizon)
            else:
                # Act upon deterministic model states
                imagined_states, action_log_probs, actions = self._model.rollout_policy_deterministic(state=start_model_states,
                                                                                                      policy_fn=policy_fn,
                                                                                                      num_steps=self._imagine_horizon)
            
            if include_first:
                # Add initial state back to imagined_states, since it is natively not added to the returned list
                imagined_states = {k: torch.cat((start_model_states[k].unsqueeze(1), v), 1) for k, v in imagined_states.items()}
                # Remove last step for consistency
                imagined_states = {k: v[:, :-1] for k, v in imagined_states.items()}
            
            return imagined_states, action_log_probs, actions
        
    def rollout_from_actions(self, pred_actions, max_length=-1, start_phys_state=None):
        """
        Interact with the environment using the predicted actions.
        """
        observations, rewards, infos = [], [], []

        # Reset environment to start physical state, if available.
        obs, info = (
            self._env.reset()
            if start_phys_state is None
            else self._env.reset_to_state(start_phys_state)
        )
            
        observations.append(obs)
        rewards.append(torch.FloatTensor([0.0]))
        infos.append(info)

        done = False
        i = 0

        while not done and (max_length < 0 or i < max_length):
            action = pred_actions[i].cpu()
            obs, reward, done, info = self._env.step(action=action)

            i += 1
            reward = torch.FloatTensor([reward])

            observations.append(obs)
            rewards.append(reward)
            infos.append(info)
   
        observations = [torch.stack([o[i] for o in observations], dim=0) for i in range(len(observations[0]))]
        rewards = torch.stack(rewards, dim=0)
        infos = stack_maybe_nested_dicts(infos, dim=0)

        return observations, rewards, infos
    
    def posterior_rollout_from_actions(self, pred_actions, max_length=-1, start_phys_state=None, start_model_state=None, sample_model=False):
        """
        Rollout model with given actions and the environment observation.
        """
        with torch.no_grad():
            observations, rewards, infos, post_states = self._posterior_rollout_from_actions(pred_actions=pred_actions,
                                                                                             max_length=max_length,
                                                                                             start_phys_state=start_phys_state,
                                                                                             start_model_state=start_model_state,
                                                                                             sample_model=sample_model)
            return observations, rewards, infos, post_states
    
    def _posterior_rollout_from_actions(self, pred_actions, max_length=-1, start_phys_state=None, start_model_state=None, sample_model=False):
        """
        Interact with the environment using the predicted actions.
        """
        observations, rewards, infos, post_states = [], [], [], []
 
        # Reset environment to start physical state, if available.
        obs, info = (
            self._env.reset()
            if start_phys_state is None
            else self._env.reset_to_state(start_phys_state)
        )
        # Use given start model state, if available.
        policy_state = (
            self._policy.get_initial(batch_size=1)[1]
            if start_model_state is None
            else start_model_state
        )
        # Use given start action from action trajectory.
        action = pred_actions[0]

        observations.append(obs)
        rewards.append(torch.FloatTensor([0.0]))
        infos.append(info)
        post_states.append(policy_state)

        done = False
        i = 0

        while not done and (max_length < 0 or i < max_length-1) and i < len(pred_actions):
            action = pred_actions[i].cpu()
            
            obs_for_pol = self._prepare_observation_for_policy(observation=obs)
            action_for_pol = self._prepare_action_for_policy(action=action)
            
            # Only predict the next model state, not the action.
            if sample_model:
                # Act upon sampled model state
                _, policy_state = self._policy(observation=obs_for_pol,
                                               prev_action=action_for_pol,
                                               policy_state=policy_state,
                                               sample=False) # greedy policy
            else:
                # Act upon deterministic model state
                _, policy_state = self._policy.act_on_deterministic_state(observation=obs_for_pol,
                                                                          prev_action=action_for_pol,
                                                                          policy_state=policy_state,
                                                                          sample=False) # greedy policy
            obs, reward, done, info = self._env.step(action=action)

            i += 1
            reward = torch.FloatTensor([reward])

            observations.append(obs)
            rewards.append(reward)
            infos.append(info)
            post_states.append(policy_state)
   
        observations = [torch.stack([o[i] for o in observations], dim=0) for i in range(len(observations[0]))]
        rewards = torch.stack(rewards, dim=0)
        infos = stack_maybe_nested_dicts(infos, dim=0)
        post_states = stack_maybe_nested_dicts(post_states, dim=0)
        post_states = {k: v.squeeze(1) for k, v in post_states.items()}

        return observations, rewards, infos, post_states
    
    def prior_rollout_from_actions(self, pred_actions, start_model_states, sample_model=False, include_first=True):
        """
        Rollout policy on priors only, which corresponds to no intergration of the environment observation into the model states.
        """
        with torch.no_grad():
            # Rollout model with given actions only
            if sample_model:
                # Predict upon sampled model states
                imagined_states = self._model.predict_states_open_loop(initial_state=start_model_states, actions=pred_actions)
            else:
                # Predict upon deterministic model states
                imagined_states = self._model.predict_states_open_loop_deterministic(initial_state=start_model_states, actions=pred_actions)
            
            if include_first:
                # Add initial state back to imagined_states, since it is natively not added to the returned list
                imagined_states = {k: torch.cat((start_model_states[k].unsqueeze(1), v), 1) for k, v in imagined_states.items()}
                # Remove last step for consistency
                imagined_states = {k: v[:, :-1] for k, v in imagined_states.items()}
                                                                                        
            return imagined_states