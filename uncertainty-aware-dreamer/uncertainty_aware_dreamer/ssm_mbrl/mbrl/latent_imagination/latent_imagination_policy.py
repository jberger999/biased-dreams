import torch
from typing import Optional
import uncertainty_aware_dreamer.ssm_mbrl.common.dense_nets as dn
from uncertainty_aware_dreamer.ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy
from uncertainty_aware_dreamer.ssm_mbrl.common.img_preprocessor import ImgPreprocessor
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.actor import Actor

nn = torch.nn


class Critic(nn.Module):

    def __init__(self,
                 in_dim,
                 num_layers: int,
                 hidden_size: int,
                 activation: str = "ReLU"):

        super(Critic, self).__init__()

        hidden_layers, size_last_hidden = dn.build_hidden_layers(in_features=in_dim,
                                                                 layer_sizes=num_layers * [hidden_size],
                                                                 activation=activation)
        hidden_layers.append(torch.nn.Linear(in_features=size_last_hidden,
                                             out_features=1))
        self._v_net = nn.Sequential(*hidden_layers)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self._v_net(in_features)


class LIPolicy(AbstractPolicy, torch.nn.Module):

    def __init__(self,
                 model: AbstractSSM,
                 actor: Actor,
                 critic: Critic,
                 action_dim: int,
                 obs_are_images: list[bool],
                 img_preprocessor: ImgPreprocessor,
                 device: torch.device,
                 mean_for_actor: bool):
        super(LIPolicy, self).__init__(model=model,
                                       action_dim=action_dim,
                                       obs_are_images=obs_are_images,
                                       img_preprocessor=img_preprocessor,
                                       device=device)
        self._mean_for_actor = mean_for_actor
        self.actor = actor
        self.critic = critic
        self.model = model

    def _call_internal(self,
                       observation: list[torch.Tensor],
                       prev_action: torch.Tensor,
                       policy_state: dict,
                       sample: bool) -> tuple[torch.Tensor, dict]:
        post_state = self.model.get_next_posterior(observation=observation,
                                                   action=prev_action,
                                                   post_state=policy_state)
        features = self.model.get_features(post_state)
        action = self.actor(features, sample)
        return action, post_state
    
    def _act_on_deterministic_state(self,
                                    observation: list[torch.Tensor],
                                    prev_action: torch.Tensor,
                                    policy_state: dict,
                                    sample: bool) -> tuple[torch.Tensor, dict]:
        """
        Instead of action on model state features, act on deterministic ones.
        """
        # Prior state prediction is done deterministically
        post_state = self.model.get_next_posterior_deterministic(observation=observation,
                                                                 action=prev_action,
                                                                 post_state=policy_state)
        features = self.model.get_deterministic_features(post_state)
        action = self.actor(features, sample)
        return action, post_state