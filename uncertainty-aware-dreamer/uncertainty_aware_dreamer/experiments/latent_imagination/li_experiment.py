from uncertainty_aware_dreamer.ssm_mbrl.mbrl.latent_imagination.latent_imagination_trainer_factory import LIPolicyTrainerFactory, UncertainLIPolicyTrainerFactory
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.latent_imagination.latent_imagination_policy_factory import LIPolicyFactory
from uncertainty_aware_dreamer.ssm_mbrl.rssm.objectives.objective_factory import AERSSMObjectiveFactory
from uncertainty_aware_dreamer.experiments.common.wm_experiment import WorldModelExperiment


class LIExperiment(WorldModelExperiment):
    
    def __init__(self, overrides_config):
        super().__init__(overrides_config)
    
    def policy_setup(self):
        policy_factory = LIPolicyFactory()
        policy_config = self._overrides_config.algorithm.policy if "policy" in self._overrides_config.algorithm else policy_factory.get_default_config()
        
        return policy_config, policy_factory
    
    def trainer_setup(self):
        # ------------------------------------------------------------
        # --------------------------- RSSM ---------------------------
        # ------------------------------------------------------------
        if self._overrides_config.algorithm.world_model.name == "rssm" or self._overrides_config.algorithm.world_model.name == "cat_rssm":
            trainer_factory = LIPolicyTrainerFactory(AERSSMObjectiveFactory())
        # ------------------------------------------------------------
        # --------------------------- URSSM --------------------------
        # ------------------------------------------------------------
        elif self._overrides_config.algorithm.world_model.name == "urssm" or self._overrides_config.algorithm.world_model.name == "cat_urssm":
            trainer_factory = UncertainLIPolicyTrainerFactory(AERSSMObjectiveFactory())
        else:
            raise NotImplementedError(f"World model \"{self._overrides_config.algorithm.world_model.name}\" not implemented.")
        
        trainer_config = self._overrides_config.algorithm.policy_trainer if "policy_trainer" in self._overrides_config.algorithm else trainer_factory.get_default_config()
        
        # Remove KL scale entry for physical state decoder, if not specified
        if not self._overrides_config.algorithm.world_model.model.use_phys_state_dec:
            trainer_config.objective.decoder_loss_scales.pop(2)
        
        return trainer_config, trainer_factory