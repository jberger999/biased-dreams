from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict
from uncertainty_aware_dreamer.ssm_mbrl.rssm.transition.r_rssm_tm import RRSSMTM
from uncertainty_aware_dreamer.ssm_mbrl.rssm.transition.cat_rssm_tm import CatRSSMTM


class TransitionFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = False) -> ConfigDict:
        config = ConfigDict()
        config.type = "r_rssm"
        config.lsd = 30
        config.rec_state_dim = 200
        config.num_layers = 1
        config.hidden_size = 300
        config.min_std = 0.1
        config.activation = "ELU"
        config.state_part_for_update = "d"
        
        # Categorical
        config.num_categoricals = 32
        config.categorical_size = 32
        config.with_obs_pre_layers = False
        config.unimix_factor = 0 # disabling mixture with uniform

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(config: ConfigDict,
              obs_sizes: list[int],
              action_dim: int):
        
        if "r_rssm" in config.type:
            return RRSSMTM(obs_sizes=obs_sizes,
                           state_dim=config.lsd,
                           action_dim=action_dim,
                           rec_state_dim=config.rec_state_dim,
                           num_layers=config.num_layers,
                           hidden_size=config.hidden_size,
                           min_std=config.min_std,
                           state_part_for_update=config.state_part_for_update,
                           activation=config.activation)
        elif "cat_rssm" in config.type:
            return CatRSSMTM(obs_sizes=obs_sizes,
                             categorical_size=config.categorical_size,
                             num_categorical=config.num_categoricals,
                             action_dim=action_dim,
                             rec_state_dim=config.rec_state_dim,
                             num_layers=config.num_layers,
                             hidden_size=config.hidden_size,
                             with_obs_pre_layers=config.with_obs_pre_layers,
                             activation=config.activation,
                             unimix_factor=config.unimix_factor)
        else:
            raise AssertionError