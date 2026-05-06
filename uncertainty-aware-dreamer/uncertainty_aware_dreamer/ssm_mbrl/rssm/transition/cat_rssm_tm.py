import torch

from uncertainty_aware_dreamer.ssm_mbrl.rssm.transition.abstract_tm import AbstractRSSMTM
import uncertainty_aware_dreamer.ssm_mbrl.common.dense_nets as dn
from uncertainty_aware_dreamer.ssm_mbrl.util.one_hot_categorical import OneHotCategoricalStraightThrough

nn = torch.nn
F = torch.nn.functional
jit = torch.jit


class CatRSSMActivation(nn.Module):

    def __init__(self,
                 categorical_size: int,
                 num_categorical: int,
                 unimix_factor: float):
        super(CatRSSMActivation, self).__init__()
        self._categorical_size = categorical_size
        self._num_categorical = num_categorical
        self._unimix_factor = unimix_factor

    def forward(self, flat_logits: torch.Tensor) -> torch.Tensor:
        logits = flat_logits.reshape(flat_logits.shape[:-1] + (self._num_categorical, self._categorical_size))
        probs = torch.softmax(logits, dim=-1)
        return self._unimix_factor / self._categorical_size + (1 - self._unimix_factor) * probs


class CatRSSMTM(AbstractRSSMTM):
    """Implementation of the Categorical World Model from DreamerV2"""

    def __init__(self,
                 obs_sizes: list[int],
                 categorical_size: int,
                 num_categorical: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 with_obs_pre_layers: bool,
                 activation: str = "ReLU",
                 unimix_factor: float = 0.01):
        super(CatRSSMTM, self).__init__(action_dim=action_dim)
        self._categorical_size = categorical_size
        self._num_categorical = num_categorical
        self._state_dim = categorical_size * num_categorical
        self._rec_state_dim = rec_state_dim
        self._with_obs_pre_layers = with_obs_pre_layers
        self._obs_sizes = obs_sizes

        assert 0 <= unimix_factor < 1, "unimix_factor must be in [0, 1)"
        self._unimix_factor = unimix_factor

        self._build_predict(action_dim=action_dim,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            activation=activation)

        self._build_update(obs_sizes=obs_sizes,
                           num_layers=num_layers,
                           hidden_size=hidden_size,
                           activation=activation)

    def _unimix_softmax_activation(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return self._unimix_factor / self._categorical_size + (1 - self._unimix_factor) * probs

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       hidden_size: int,
                       activation: str):
        pre_layers, pre_last_hidden_size = dn.build_hidden_layers(in_features=self._state_dim + action_dim,
                                                                  layer_sizes=[hidden_size] * num_layers,
                                                                  activation=activation)
        self._pred_pre_layers = nn.Sequential(*pre_layers)
        self._pred_tm_cell = nn.GRUCell(input_size=pre_last_hidden_size,
                                        hidden_size=self._rec_state_dim)

        post_layers, post_last_hidden_size = dn.build_hidden_layers(in_features=self._rec_state_dim,
                                                                    layer_sizes=[hidden_size] * num_layers,
                                                                    activation=activation)
        self._pred_post_layers = nn.Sequential(*post_layers,
                                               nn.Linear(in_features=post_last_hidden_size,
                                                         out_features=self._state_dim),
                                               CatRSSMActivation(categorical_size=self._categorical_size,
                                                                 num_categorical=self._num_categorical,
                                                                 unimix_factor=self._unimix_factor))

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      hidden_size: int,
                      activation: str):
        if self._with_obs_pre_layers:
            assert len(obs_sizes) == 1, "Only one observation supported for now"
            obs_layers, obs_last_hidden_size = dn.build_hidden_layers(in_features=obs_sizes[0],
                                                                      layer_sizes=[hidden_size] * num_layers,
                                                                      activation=activation)
            self._obs_pre_net = nn.Sequential(*obs_layers,
                                              nn.LayerNorm(obs_last_hidden_size))
            inpt_size = obs_last_hidden_size + self._rec_state_dim
        else:
            inpt_size = sum(obs_sizes) + self._rec_state_dim
            self._obs_pre_net = None
        obs_layers, obs_out_size = dn.build_hidden_layers(in_features=inpt_size,
                                                          layer_sizes=[hidden_size] * num_layers,
                                                          activation=activation)
        self._updt_dist_layer = nn.Sequential(*obs_layers,
                                              nn.Linear(in_features=obs_out_size, out_features=self._state_dim),
                                              CatRSSMActivation(categorical_size=self._categorical_size,
                                                                num_categorical=self._num_categorical,
                                                                unimix_factor=self._unimix_factor))

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        probs = self._updt_dist_layer(torch.cat(obs + [prior_state["gru_cell_state"]], dim=-1))
        return {"probs": probs,
                "sample": OneHotCategoricalStraightThrough.rsample(probs),
                "gru_cell_state": prior_state["gru_cell_state"]}

    @jit.script_method
    def predict(self, post_state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        trans_input = torch.cat([post_state["sample"].flatten(start_dim=-2, end_dim=-1), action], dim=-1)
        cell_state = self._pred_tm_cell(input=self._pred_pre_layers(trans_input),
                                        hx=post_state["gru_cell_state"])
        probs = self._pred_post_layers(cell_state)
        return {"probs": probs,
                "sample": OneHotCategoricalStraightThrough.rsample(probs),
                "gru_cell_state": cell_state}
        
    @jit.script_method
    def predict_deterministic(self, post_state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict(post_state=post_state, action=action)

    @jit.script_method
    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        p = self._pred_tm_cell.weight_ih
        return {"probs": torch.zeros(size=[batch_size, self._num_categorical, self._categorical_size],
                                     dtype=p.dtype, device=p.device),
                "sample": torch.zeros(size=[batch_size, self._num_categorical, self._categorical_size],
                                      dtype=p.dtype, device=p.device),
                "gru_cell_state": torch.zeros(size=[batch_size, self._rec_state_dim], dtype=p.dtype, device=p.device)}

    @property
    def feature_size(self) -> int:
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
        flat_sample = state["sample"].reshape(*(state["sample"].shape[:-2] + (self._state_dim, )))
        return torch.cat([flat_sample, state["gru_cell_state"]], dim=-1)
    
    def get_deterministic_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        # Get mode from probs instead of sample
        # NOTE: You cannot backpropagate through these!
        flat_mode = OneHotCategoricalStraightThrough.mode(state["probs"])
        flat_mode = flat_mode.reshape(*(flat_mode.shape[:-2] + (self._state_dim, )))
        return torch.cat([flat_mode, state["gru_cell_state"]], dim=-1)
    
    def get_stochastic(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["sample"].reshape(*(state["sample"].shape[:-2] + (self._state_dim, )))
    
    def get_deterministic(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["gru_cell_state"]
    
    def repeat_state(self, state: dict[str, torch.Tensor], num_repeat: int = 1) -> torch.Tensor:
        state_repeat = {}
        if len(state["sample"].shape) == 3:
            # state = [1 x dims*]
            state_repeat["probs"] = state["probs"].repeat(num_repeat, 1, 1)
            state_repeat["sample"] = state["sample"].repeat(num_repeat, 1, 1)
            state_repeat["gru_cell_state"] = state["gru_cell_state"].repeat(num_repeat, 1)
        elif len(state["sample"].shape) == 4:
            # state = [1 x seq_len x dims*]
            state_repeat["probs"] = state["probs"].repeat(num_repeat, 1, 1, 1)
            state_repeat["sample"] = state["sample"].repeat(num_repeat, 1, 1, 1)
            state_repeat["gru_cell_state"] = state["gru_cell_state"].repeat(num_repeat, 1, 1)
        else:
            raise NotImplementedError
        return state_repeat

    @property
    def latent_distribution(self) -> str:
        return "categorical"