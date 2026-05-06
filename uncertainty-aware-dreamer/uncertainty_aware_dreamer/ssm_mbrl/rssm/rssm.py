import torch
from typing import Optional, Callable

from uncertainty_aware_dreamer.ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from uncertainty_aware_dreamer.ssm_mbrl.rssm.transition.abstract_tm import AbstractRSSMTM

nn = torch.nn


class RSSM(AbstractSSM):

    def __init__(self,
                 encoders: nn.ModuleList,
                 transition_model: AbstractRSSMTM,
                 decoders: nn.ModuleList):

        super(RSSM, self).__init__()

        self.encoders = encoders
        self.transition_model = transition_model
        self.decoders = decoders

    def encode(self,
               observations: list[torch.Tensor]) -> list[torch.Tensor]:
        embedded_obs = [enc(observations[i]) for i, enc in enumerate(self.encoders)]
        return embedded_obs

    @property
    def feature_size(self) -> int:
        return self.transition_model.feature_size

    @property
    def action_dim(self) -> int:
        return self.transition_model.action_dim

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transition_model.get_features(state)
    
    def get_deterministic_features(self, state):
        return self.transition_model.get_deterministic_features(state)
    
    def get_stochastic(self, state):
        return self.transition_model.get_stochastic(state)
    
    def get_deterministic(self, state):
        return self.transition_model.get_deterministic(state)
    
    def repeat_state(self, state, num_repeat):
        return self.transition_model.repeat_state(state, num_repeat)

    def get_initial_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self.transition_model.get_initial(batch_size=batch_size)

    @property
    def latent_distribution(self):
        return self.transition_model.latent_distribution

    def get_next_posterior(self,
                           observation: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict) -> dict:
        observation = [torch.unsqueeze(obs, dim=1) for obs in observation]
        latent_obs = [lo.squeeze(dim=1) for lo in self.encode(observation)]
        return self.transition_model.get_next_posterior(latent_obs=latent_obs,
                                                        action=action,
                                                        post_state=post_state)
        
    def get_next_posterior_deterministic(self,
                                         observation: list[torch.Tensor],
                                         action: torch.Tensor,
                                         post_state: dict) -> dict:
        observation = [torch.unsqueeze(obs, dim=1) for obs in observation]
        latent_obs = [lo.squeeze(dim=1) for lo in self.encode(observation)]
        return self.transition_model.get_next_posterior_deterministic(latent_obs=latent_obs,
                                                                      action=action,
                                                                      post_state=post_state)

    def predict_rewards_open_loop(self,
                                  initial_state: dict,
                                  actions: torch.Tensor):
        states = self.transition_model.open_loop_prediction(initial_state=initial_state,
                                                            actions=actions)
        return self.decoders[-1](self.transition_model.get_features(state=states))[0]

    def rollout_policy(self,
                       state: dict,
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict, torch.Tensor]:
        return self.transition_model.rollout_policy(state=state,
                                                    policy_fn=policy_fn,
                                                    num_steps=num_steps)
        
    def rollout_policy_deterministic(self,
                                     state: dict,
                                     policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                                     num_steps: int) -> tuple[dict, torch.Tensor]:
        return self.transition_model.rollout_policy_deterministic(state=state,
                                                                  policy_fn=policy_fn,
                                                                  num_steps=num_steps)
        
    def predict_states_open_loop(self,
                                 initial_state: dict,
                                 actions: torch.Tensor):
        return self.transition_model.open_loop_prediction(initial_state=initial_state,
                                                          actions=actions)
        
    
    def predict_states_open_loop_deterministic(self,
                                               initial_state: dict,
                                               actions: torch.Tensor):
        return self.transition_model.open_loop_prediction_deterministic(initial_state=initial_state,
                                                                        actions=actions)