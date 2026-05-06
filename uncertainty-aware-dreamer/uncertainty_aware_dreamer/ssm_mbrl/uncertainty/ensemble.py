from typing import Tuple
from collections import OrderedDict
import torch
import numpy as np

from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict
import uncertainty_aware_dreamer.ssm_mbrl.common.dense_nets as dn
import uncertainty_aware_dreamer.ssm_mbrl.common.modules as mod

nn = torch.nn
opt = torch.optim

LOG_SQRT_2_PI = np.log(np.sqrt(2 * np.pi))


class EnsembleModelFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True):
        config = ConfigDict()
        
        # Architecture
        config.num_ensemble = 5
        config.hidden_size = 300    # for consistency with RSSM
        config.activation = "ELU"   # for consistency with RSSM
        config.num_layers = 2
        config.init_std = 1.0       # guessed value
        config.min_std = 0.1        # for consistency with RSSM
        config.max_std = 5.0        # ignored if not sigmoid_activation
        config.sigmoid_activation = False
        config.output_normalization = "none"
        
        if finalize_adding:
            config.finalize_adding()
            
        return config


    @staticmethod
    def build(world_model,
              config: ConfigDict) -> Tuple[nn.Module, int]:

        return EnsembleModel(feature_size=world_model.transition_model.feature_size,
                             stoch_size=world_model.transition_model.stoch_size,
                             deter_size=world_model.transition_model.deter_size,
                             embed_size=world_model.transition_model.obs_sizes[0],
                             action_size=world_model.transition_model.action_size,
                             dyn_discrete=(world_model.transition_model.latent_distribution == "categorical"),
                             config=config)
    
    
class EnsembleModel(nn.Module):
    
    def __init__(self,
                 feature_size: int,
                 stoch_size: int,
                 deter_size: int,
                 embed_size: int,
                 action_size: int,
                 dyn_discrete: bool,
                 config: ConfigDict):
        
        super(EnsembleModel, self).__init__()
        
        # Define input and output sizes according to world model transitions.
        self._input_size = feature_size + action_size
        self._output_size = {
            "stoch": stoch_size,
            "deter": deter_size,
            "feat": feature_size,
            "embed": embed_size,
        }[config.target]
        self._target = config.target
        self._dyn_discrete = dyn_discrete
        
        # Build specified number of ensemble members.
        self._num_ensemble = config.num_ensemble
        self._models = nn.ModuleList([self._build_ensemble_member(self._input_size, self._output_size, config) for _ in range(config.num_ensemble)])
        
        # Define disagreement measure.
        if config.distance_measure == "gjs":
            from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.distance_measure import GeometricJensenShannonDivergence
            self._distance_measure = GeometricJensenShannonDivergence()
        elif config.distance_measure == "jrd":
            from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.distance_measure import JensenRenyiDivergence
            self._distance_measure = JensenRenyiDivergence()
        else:
            raise NotImplementedError("Distance measure " + config.distance_measure + " not defined for ensemble disagreement.")  
        

    def _build_ensemble_member(self, 
                               input_size: int, 
                               output_size: int,
                               config: ConfigDict) -> nn.Module:
        """
        Build a single ensemble member.
        A single ensemble member is a 2 hidden-layer MLP, which takes in the RNN-state of RSSM and the action as inputs,
        and predicts the encoder features as a Gaussian distribution.
        """
        # Build simple MLP with hidden layers and chosen activation.
        pre_layers, hidden_out_size = dn.build_hidden_layers(in_features=input_size,
                                                             layer_sizes=[config.hidden_size] * config.num_layers,
                                                             activation=config.activation,
                                                             normalization=config.normalization)
        
        # Defines output layers, that separates mean and variance, and applies a softplus activation to the variance.
        post_layers = mod.DiagonalGaussianParameterLayer(in_features=hidden_out_size,
                                                         distribution_dim=output_size,
                                                         init_var=config.init_std,
                                                         min_var=config.min_std,
                                                         max_var=config.max_std,
                                                         sigmoid_activation=config.sigmoid_activation,
                                                         output_normalization=config.output_normalization)

        return nn.Sequential(*pre_layers, post_layers)
    

    def forward(self, 
                state: torch.Tensor,
                action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through all ensemble members.
        Member predictions are stacked along dim=0.
        """
        x = torch.cat([state, action], dim=-1)
        l = [self._forward_single(x, model) for model in self._models]
        mean = torch.stack([pred["mean"] for pred in l], dim=0)
        std = torch.stack([pred["std"] for pred in l], dim=0)
        return {"mean": mean, "std": std}


    def _forward_single(self, 
                        x: torch.Tensor,
                        model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Forward pass through a single ensemble member.
        """
        mean, std = model(x)
        return {"mean": mean, "std": std}


    def get_ensemble_target(self, model, state, embed_obs):
        target = {
            "stoch": model.get_stochastic(state),
            "deter": model.get_deterministic(state),
            "feat": model.get_deterministic_features(state), # TODO: sampled feature possibly sufficient
            "embed": embed_obs[0]
        }
        return target[self._target]


    def _gaussian_nll(self,
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        if len(mean.shape) > len(target.shape):
            target = target.unsqueeze(0).expand(mean.shape)
        
        norm_term = - std.log() - LOG_SQRT_2_PI
        exp_term = - 0.5 * (target - mean).square() / std.square()

        # Sum over output dimensions and negate to get NLL.
        sample_wise_nll = - (norm_term + exp_term).sum(-1)
        
        # Additionally compute MSE to log.
        with torch.no_grad():
            sample_wise_mse = (target - mean).square().mean(-1)
        
        return sample_wise_nll, sample_wise_mse


    def _reconstruction_loss(self,
                             mean: torch.Tensor,
                             std: torch.Tensor,
                             target: torch.Tensor,
                             masking_prob: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Compute Gaussian neg. log-likelihood (and MSE)
        nll, mse = self._gaussian_nll(mean, std, target)
        
        # Mask random sequences, so every ensemble member only sees a subset of them.
        num_seq, len_seq = nll.shape[1:]
        mask = torch.zeros(self._num_ensemble, num_seq, 1, device=nll.device)
        for i in range(self._num_ensemble):
            seq_idx = torch.randperm(num_seq)[:int(masking_prob * num_seq)]
            mask[i, seq_idx, 0] = 1.0
        mask = mask.expand(-1, -1, len_seq)
        
        # Proper normalization so masking does not shrink the loss
        return (nll * mask).sum() / mask.sum(), (mse * mask).sum() / mask.sum()


    def compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     targets: torch.Tensor) -> tuple[torch.Tensor, dict]: 
        
        # Assumes that states, actions, and targets are accurately shifted.
        pred = self(state=states, action=actions)
        nll, mse = self._reconstruction_loss(mean=pred["mean"],
                                             std=pred["std"],
                                             target=targets)
        
        
        log_dict = OrderedDict({"ensemble_nll": nll.detach().item(),
                                "ensemble_mse": mse.detach().item()})
        
        return nll, log_dict


    @torch.no_grad()
    def compute_disagreement(self,
                             state: torch.Tensor,
                             action: torch.Tensor) -> torch.Tensor:
        
        self._models.eval()
        
        # Assumes that states, actions, and targets are accurately shifted.
        ensemble_preds = self(state=state, action=action)
        
        return self._distance_measure.compute_measure(means=ensemble_preds["mean"],
                                                      vars=ensemble_preds["std"].square())