import torch

import uncertainty_aware_dreamer.experiments.dmc_common.hidden_layers as dmc_archs
from uncertainty_aware_dreamer.envs.env_factory import DMCEnvFactory, StateBasedDMCEnvFactory
from uncertainty_aware_dreamer.ssm_mbrl.rssm.rssm_factory import RSSMFactory
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.mbrl_experiment import MBRLExperiment
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.mbrl_factory import MBRLFactory
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.evaluation.reward_eval_factory import RewardEvalFactory
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.ensemble import EnsembleModelFactory

nn = torch.nn


class WorldModelExperiment:
    
    def __init__(self, overrides_config):
        self._overrides_config = overrides_config
    
    def policy_setup(self):
        raise NotImplementedError
    
    def trainer_setup(self):
        raise NotImplementedError
    
    def model_setup(self):
        encoder_factories, decoder_factories = dmc_archs.get_standard_ae_mujoco_factories(obs_type=self._overrides_config.environment.env.obs_type)
        
        # Remove physical state decoder, if not specified
        if not self._overrides_config.algorithm.world_model.model.use_phys_state_dec:
            decoder_factories.pop(2)
            
        model_factory = RSSMFactory(encoder_factories=encoder_factories,
                                    decoder_factories=decoder_factories)

        model_config = self._overrides_config.algorithm.world_model.model if "world_model" in self._overrides_config.algorithm else model_factory.get_default_config()
        
        return model_config, model_factory
    
    def ensemble_setup(self):
        # ------------------------------------------------------------
        # --------------------------- URSSM --------------------------
        # ------------------------------------------------------------
        if self._overrides_config.algorithm.world_model.name == "urssm" or self._overrides_config.algorithm.world_model.name == "cat_urssm":
            ensemble_factory = EnsembleModelFactory()
            ensemble_config = self._overrides_config.algorithm.ensemble if "ensemble" in self._overrides_config.algorithm else ensemble_factory.get_default_config()
        else:
            ensemble_factory = None
            ensemble_config = None
        return ensemble_config, ensemble_factory
    
    def create_experiment(self, verbose=True):
        # -----------------------------------------------------------
        # ----------------------- ENVIRONMENT -----------------------
        # -----------------------------------------------------------
        env_config = self._overrides_config.environment.env if "environment" in self._overrides_config else DMCEnvFactory().get_default_config()
        
        # -----------------------------------------------------------
        # ------------------------ MODEL ----------------------------
        # -----------------------------------------------------------
        model_config, model_factory = self.model_setup()
        
        # -----------------------------------------------------------
        # ---------------------- ENSEMBLE ---------------------------
        # -----------------------------------------------------------
        ensemble_config, ensemble_factory = self.ensemble_setup()

        # -----------------------------------------------------------
        # ------------------------ POLICY ---------------------------
        # -----------------------------------------------------------
        policy_config, policy_factory = self.policy_setup()
        
        # -----------------------------------------------------------
        # ----------------------- TRAINER ---------------------------
        # -----------------------------------------------------------
        trainer_config, trainer_factory = self.trainer_setup()

        # -----------------------------------------------------------
        # ------------------------ MBRL -----------------------------
        # -----------------------------------------------------------
        mbrl_config = self._overrides_config.algorithm.mbrl if "mbrl" in self._overrides_config.algorithm else MBRLFactory().get_default_config()
        
        # -----------------------------------------------------------
        # --------------------- REWARD EVAL ------------------------
        # -----------------------------------------------------------
        reward_eval_config = self._overrides_config.algorithm.reward_eval if "reward_eval" in self._overrides_config.algorithm else RewardEvalFactory().get_default_config()
        
        # -----------------------------------------------------------
        # --------------------- EXPERIMENT --------------------------
        # -----------------------------------------------------------
        experiment = MBRLExperiment(env_factory=StateBasedDMCEnvFactory,
                                    model_factory=model_factory,
                                    policy_factory=policy_factory,
                                    trainer_factory=trainer_factory,
                                    mbrl_factory=MBRLFactory(),
                                    ensemble_factory=ensemble_factory,
                                    policy_eval_factories=[RewardEvalFactory()],
                                    verbose=verbose,
                                    seed=self._overrides_config.seed,
                                    use_cuda_if_available=True,
                                    fully_deterministic=self._overrides_config.deterministic)

        # Final build according to the configuration
        experiment.build(env_config=env_config,
                         model_config=model_config,
                         ensemble_config=ensemble_config,
                         policy_config=policy_config,
                         trainer_config=trainer_config,
                         mbrl_config=mbrl_config,
                         policy_eval_config={"reward_eval": reward_eval_config})
        
        # Additional infos for main
        infos = {
            "data_collect_seq": mbrl_config.data_collection.num_sequences,
            "init_data_collect_seq": mbrl_config.initial_data_collection.num_sequences,
        }
        
        return experiment, infos