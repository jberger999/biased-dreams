import torch
from typing import Optional, Callable

import time

nn = torch.nn
jit = torch.jit
F = torch.nn.functional
dists = torch.distributions


class AbstractRSSMTM(jit.ScriptModule):

    def __init__(self,
                 action_dim: int):
        super(AbstractRSSMTM, self).__init__()
        self._action_dim = action_dim
        self._default_value = 0.0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @staticmethod
    def _stack_dicts(dicts: list[dict[str, torch.Tensor]], dim: int = 1) -> dict[str, torch.Tensor]:
        return {k: torch.stack([d[k] for d in dicts], dim=dim) for k in dicts[0].keys()}

    @jit.script_method
    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @jit.script_method
    def predict_deterministic(self,
                              post_state: dict[str, torch.Tensor],
                              action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @jit.script_method
    def _predict_for_rollout(self,
                             post_state: dict[str, torch.Tensor],
                             action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict(post_state=post_state, action=action)
    
    @jit.script_method
    def _predict_for_rollout_deterministic(self,
                                           post_state: dict[str, torch.Tensor],
                                           action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict_deterministic(post_state=post_state, action=action)

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @jit.script_method
    def full_update(self,
                    prior_state: dict[str, torch.Tensor],
                    obs: list[torch.Tensor],
                    target_obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @jit.script_method
    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @jit.script_method
    def sample_gauss(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(mean) * std + mean

    @jit.script_method
    def sample_cat_straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from the categorical distribution, using straight-through gradients"""
        #return dists.OneHotCategoricalStraightThrough(logits=logits, validate_args=False).rsample()
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        probs_2d = probs.reshape(-1, probs.shape[-1])
        samples_2d = torch.multinomial(input=probs_2d, num_samples=1, replacement=True)
        samples_2d = torch.nn.functional.one_hot(samples_2d, num_classes=probs.shape[-1]).to(probs)
        samples = samples_2d.view(probs.shape)
        samples = samples + (probs - probs.detach())
        return samples

    @property
    def feature_size(self):
        raise NotImplementedError

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def get_deterministic_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def get_stochastic(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def get_deterministic(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def repeat_state(self, state: dict[str, torch.Tensor], num_repeat: int = 1) -> torch.Tensor:
        raise NotImplementedError

    #######################
    # Inference Factories #
    #######################

    def get_next_posterior(self,
                           latent_obs: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict) -> dict[str, torch.Tensor]:

        prior_state = self.predict(post_state=post_state, action=action)
        post_state = self.update(prior_state=prior_state, obs=latent_obs)
        return post_state
    
    def get_next_posterior_deterministic(self,
                                         latent_obs: list[torch.Tensor],
                                         action: torch.Tensor,
                                         post_state: dict) -> dict[str, torch.Tensor]:

        prior_state = self.predict_deterministic(post_state=post_state, action=action)
        post_state = self.update(prior_state=prior_state, obs=latent_obs)
        return post_state

    def forward_pass(self,
                     embedded_obs: list[torch.Tensor],
                     actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        batch_size, seq_length = embedded_obs[0].shape[:2]
        embedded_obs = [torch.unbind(lo, 1) for lo in embedded_obs]
        actions = torch.unbind(actions, 1)

        post_state = self.get_initial(batch_size=batch_size)

        prior_states = jit.annotate(list[dict[str, torch.Tensor]], [])
        post_states = jit.annotate(list[dict[str, torch.Tensor]], [])
        for i in range(seq_length):
            prior_state = self.predict(post_state=post_state,
                                       action=actions[i])
            post_state = self.update(prior_state=prior_state,
                                     obs=[lo[i] for lo in embedded_obs])

            prior_states.append(prior_state)
            post_states.append(post_state)
        return self._stack_dicts(post_states), self._stack_dicts(prior_states)
        
    @torch.jit.script_method
    def open_loop_prediction(self,  initial_state: dict[str, torch.Tensor], actions: torch.Tensor)\
            -> dict[str, torch.Tensor]:
        state = initial_state
        action_list = torch.unbind(actions, 1)
        states = torch.jit.annotate(list[dict[str, torch.Tensor]], [])
        for i, action in enumerate(action_list):
            state = self._predict_for_rollout(post_state=state,
                                              action=action)
            states.append(state)

        return self._stack_dicts(states)
    
    @torch.jit.script_method
    def open_loop_prediction_deterministic(self, initial_state: dict[str, torch.Tensor], actions: torch.Tensor)\
            -> dict[str, torch.Tensor]:
        """
        Rollout model with given action list deterministically.
        """
        state = initial_state
        action_list = torch.unbind(actions, 1)
        states = torch.jit.annotate(list[dict[str, torch.Tensor]], [])
        for i, action in enumerate(action_list):
            state = self._predict_for_rollout_deterministic(post_state=state,
                                                            action=action)
            states.append(state)

        return self._stack_dicts(states)

    def rollout_policy(self,
                       state: dict,
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict, torch.Tensor]:
        states = []
        log_probs = []
        actions = []
        for _ in range(num_steps):
            action, log_prob = policy_fn(self.get_features(state))
            state = self._predict_for_rollout(post_state=state, action=action)

            states.append(state)
            log_probs.append(log_prob)
            actions.append(action)
        states = {k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()}
        return states, torch.stack(log_probs, dim=1), torch.stack(actions, dim=1)
    
    def rollout_policy_deterministic(self,
                                     state: dict,
                                     policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                                     num_steps: int) -> tuple[dict, torch.Tensor]:
        """
        Rollout model with given policy function deterministically.
        """
        states = []
        log_probs = []
        actions = []
        for _ in range(num_steps):
            action, log_prob = policy_fn(self.get_deterministic_features(state))
            state = self._predict_for_rollout_deterministic(post_state=state, action=action)

            states.append(state)
            log_probs.append(log_prob)
            actions.append(action)
        states = {k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()}
        return states, torch.stack(log_probs, dim=1), torch.stack(actions, dim=1)

    @property
    def latent_distribution(self) -> str:
        raise NotImplementedError
