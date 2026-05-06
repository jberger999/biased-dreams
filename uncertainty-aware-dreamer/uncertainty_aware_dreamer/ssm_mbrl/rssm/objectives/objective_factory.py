from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict
from uncertainty_aware_dreamer.ssm_mbrl.rssm.objectives.autoencoding_vi_objective import AEVIObjective
from uncertainty_aware_dreamer.ssm_mbrl.rssm.objectives.kl_objective import RSSMKLObjective

class KLRSSMObjectiveFactory:
    
    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.decoder_loss_scales = [1.0]

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(model,
              kl_objective,
              config: ConfigDict):
        return AEVIObjective(rssm=model,
                             kl_objective=kl_objective,
                             decoder_loss_scales=config.decoder_loss_scales)

class AERSSMObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.decoder_loss_scales = [1.0]

        config.kl.scale_factor = 1.0
        config.kl.free_nats = 3.0
        config.kl.balanced = False
        config.kl.alpha = 0.8

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(model,
              config: ConfigDict):
        
        kl_objective = RSSMKLObjective(distribution=model.transition_model.latent_distribution,
                                       scale_factor=config.kl.scale_factor,
                                       free_nats=config.kl.free_nats,
                                       balanced=config.kl.balanced,
                                       alpha=config.kl.alpha)
        return AEVIObjective(rssm=model,
                             kl_objective=kl_objective,
                             decoder_loss_scales=config.decoder_loss_scales)