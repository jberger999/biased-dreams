import torch
from typing import Optional
import uncertainty_aware_dreamer.ssm_mbrl.common.modules as mod
import uncertainty_aware_dreamer.ssm_mbrl.common.dense_nets as dn
from uncertainty_aware_dreamer.ssm_mbrl.rssm.transition.abstract_tm import AbstractRSSMTM

nn = torch.nn
jit = torch.jit


class RRSSMTM(AbstractRSSMTM):

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 min_std: float,
                 state_part_for_update: str = "d",  # either "d(eterministic)", "s(tochastic)" or "b(oth)"
                 activation: str = "ReLU"):
        
        super(RRSSMTM, self).__init__(action_dim=action_dim)
        self._state_dim = state_dim
        self._rec_state_dim = rec_state_dim
        self._state_part_for_update = state_part_for_update
        self._min_std = min_std
        self._obs_sizes = obs_sizes

        if self._state_part_for_update.lower() in ["e", "estem", "estimate"]:
            self._prior_input_size = 2 * self._state_dim
        elif self._state_part_for_update.lower() in ["s", "stoch", "sample"]:
            self._prior_input_size = self._state_dim
        elif self._state_part_for_update.lower() in ["d", "det", "deterministic"]:
            self._prior_input_size = self._rec_state_dim
        elif self._state_part_for_update.lower() in ["b", "both"]:
            self._prior_input_size = 2 * self._state_dim + self._rec_state_dim
        else:
            raise AssertionError

        self._build_predict(action_dim=action_dim,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            activation=activation,
                            min_std=min_std)

        self._build_update(obs_sizes=obs_sizes,
                           num_layers=num_layers,
                           hidden_size=hidden_size,
                           activation=activation,
                           min_std=min_std)

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       hidden_size: int,
                       activation: str,
                       min_std: float):
        
        pre_layers, pre_hidden_out_size = dn.build_hidden_layers(in_features=self._state_dim + action_dim,
                                                                 layer_sizes=[hidden_size] * num_layers,
                                                                 activation=activation)
        self._pred_pre_hidden_layers = nn.Sequential(*pre_layers)
        self._pred_tm_cell = nn.GRUCell(input_size=pre_hidden_out_size,
                                        hidden_size=self._rec_state_dim)

        post_layers, post_hidden_out_size = dn.build_hidden_layers(in_features=self._rec_state_dim,
                                                                   layer_sizes=[hidden_size] * num_layers,
                                                                   activation=activation)
        post_layers.append(mod.SimpleGaussianParameterLayer(in_features=post_hidden_out_size,
                                                            distribution_dim=self._state_dim,
                                                            min_std_or_var=min_std))
        self._pred_post_layers = nn.Sequential(*post_layers)

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      hidden_size: int,
                      activation: str,
                      min_std: float):

        inpt_size = sum(obs_sizes) + self._prior_input_size
        hidden_layers, hidden_out_size = dn.build_hidden_layers(in_features=inpt_size,
                                                                layer_sizes=[hidden_size] * num_layers,
                                                                activation=activation)
        hidden_layers.append(mod.SimpleGaussianParameterLayer(in_features=hidden_out_size,
                                                              distribution_dim=self._state_dim,
                                                              min_std_or_var=min_std))
        self._updt_dist_layer = nn.Sequential(*hidden_layers)

    @jit.script_method
    def _get_prior_input(self, prior_state: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        if self._state_part_for_update.lower() in ["e", "estem", "estimate"]:
            return [prior_state["mean"], prior_state["std"]]
        elif self._state_part_for_update.lower() in ["s", "stoch", "sample"]:
            return [prior_state["sample"]]
        elif self._state_part_for_update.lower() in ["d", "det", "deterministic"]:
            return [prior_state["gru_cell_state"]]
        elif self._state_part_for_update.lower() in ["b", "both"]:
            return [prior_state["mean"], prior_state["std"], prior_state["gru_cell_state"]]
        else:
            raise AssertionError

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pm, ps = self._updt_dist_layer(torch.cat(obs + self._get_prior_input(prior_state=prior_state), dim=-1))
        return {"mean": pm,
                "std": ps,
                "sample": self.sample_gauss(pm, ps),
                "gru_cell_state": prior_state["gru_cell_state"]}

    @jit.script_method
    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Perform a single step prediction of the next state given the current state, GRU state, and an action.

            Input:  posterior state, GRU state, action
            Output: next prior state as mean and variance, next GRU state
        """
        trans_input = post_state["sample"] if action is None else torch.cat([post_state["sample"], action], dim=-1)
        cell_state = self._pred_tm_cell(self._pred_pre_hidden_layers(trans_input),
                                        post_state["gru_cell_state"])
        m, s = self._pred_post_layers(cell_state)
        return {"mean": m,
                "std": s,
                "sample": self.sample_gauss(m, s),
                "gru_cell_state": cell_state}
    
    @jit.script_method
    def predict_deterministic(self,
                              post_state: dict[str, torch.Tensor],
                              action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Predict next state deterministically.

            Input:  posterior state, GRU state, action
            Output: next prior state as mean and variance, next GRU state
        """
        trans_input = post_state["mean"] if action is None else torch.cat([post_state["mean"], action], dim=-1)
        cell_state = self._pred_tm_cell(self._pred_pre_hidden_layers(trans_input),
                                        post_state["gru_cell_state"])
        m, s = self._pred_post_layers(cell_state)
        return {"mean": m,
                "std": s,
                "sample": self.sample_gauss(m, s),
                "gru_cell_state": cell_state}

    @jit.script_method
    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        p = self._pred_tm_cell.weight_ih
        return {"mean": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                # this should never be used in the rssm implementation
                "std": - torch.ones(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "sample": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "gru_cell_state": torch.zeros(size=[batch_size, self._rec_state_dim],
                                              device=p.device,
                                              dtype=p.dtype)}

    @property
    def feature_size(self):
        return self._state_dim + self._rec_state_dim
    
    @property
    def stoch_size(self):
        return self._state_dim
    
    @property
    def deter_size(self):
        return self._rec_state_dim
    
    @property
    def action_size(self):
        return self._action_dim
    
    @property
    def obs_sizes(self):
        return self._obs_sizes

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["sample"], state["gru_cell_state"]], dim=-1)
    
    def get_deterministic_features(self, state: dict[str, torch.Tensor]):
        return torch.cat([state["mean"], state["gru_cell_state"]], dim=-1)
    
    def get_stochastic(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["mean"]
    
    def get_deterministic(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["gru_cell_state"]
    
    def repeat_state(self, state: dict[str, torch.Tensor], num_repeat: int = 1) -> torch.Tensor:
        if len(state["sample"].shape) == 2:
            # state = [1 x dims*]
            return {k: v.repeat(num_repeat, 1) for k, v in state.items()}
        elif len(state["sample"].shape) == 3:
            # state = [1 x seq_len x dims*]
            return {k: v.repeat(num_repeat, 1, 1) for k, v in state.items()}
        else:
            raise NotImplementedError

    @property
    def latent_distribution(self) -> str:
        return "gaussian"
