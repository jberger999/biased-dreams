from typing import Optional

from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.latent_imagination.latent_imagination_policy import LIPolicy
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.latent_imagination.latent_imagination_trainer import LIPolicyTrainer, UncertainLIPolicyTrainer
from uncertainty_aware_dreamer.ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.ensemble import EnsembleModel


class LIPolicyTrainerFactory:

    def __init__(self,
                 objective_factory):
        self._objective_factory = objective_factory

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.lambda_ = 0.95
        config.discount = 0.99

        config.model_learning_rate = 6e-4
        config.model_adam_epsilon = 1e-8
        config.model_clip_by_norm = True
        config.model_clip_norm = 100.0
        config.model_weight_decay = 0.0

        config.actor_learning_rate = 8e-5
        config.actor_adam_epsilon = 1e-8
        config.actor_clip_by_norm = True
        config.actor_clip_norm = 100.0
        config.actor_weight_decay = 0.0

        config.critic_learning_rate = 8e-5
        config.critic_adam_epsilon = 1e-8
        config.critic_clip_by_norm = True
        config.critic_clip_norm = 100.0
        config.critic_weight_decay = 0.0

        config.imagine_horizon = 15

        config.entropy_bonus = 0.0

        config.add_subconf(name="objective",
                           sub_conf=self._objective_factory.get_default_config(finalize_adding=finalize_adding))

        config.eval_interval = -1

        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              policy: LIPolicy,
              model: AbstractSSM,
              ensemble: Optional[EnsembleModel],
              config: ConfigDict) -> LIPolicyTrainer:

        model_objective = self._objective_factory.build(model=model,
                                                        config=config.objective)

        return LIPolicyTrainer(policy=policy,
                               model_objective=model_objective,
                               lambda_=config.lambda_,
                               discount=config.discount,
                               imagine_horizon=config.imagine_horizon,
                               model_learning_rate=config.model_learning_rate,
                               model_adam_eps=config.model_adam_epsilon,
                               model_clip_norm=config.model_clip_norm,
                               model_weight_decay=config.model_weight_decay,
                               actor_learning_rate=config.actor_learning_rate,
                               actor_adam_eps=config.actor_adam_epsilon,
                               actor_clip_norm=config.actor_clip_norm,
                               actor_weight_decay=config.actor_weight_decay,
                               critic_learning_rate=config.critic_learning_rate,
                               critic_adam_eps=config.critic_adam_epsilon,
                               critic_clip_norm=config.critic_clip_norm,
                               critic_weight_decay=config.critic_weight_decay,
                               entropy_bonus=config.entropy_bonus,
                               eval_interval=config.eval_interval)
        
        
class UncertainLIPolicyTrainerFactory(LIPolicyTrainerFactory):
    
    def __init__(self, 
                 objective_factory):
        super().__init__(objective_factory)
        
    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.lambda_ = 0.95
        config.discount = 0.99

        config.model_learning_rate = 6e-4
        config.model_adam_epsilon = 1e-8
        config.model_clip_by_norm = True
        config.model_clip_norm = 100.0
        config.model_weight_decay = 0.0

        config.actor_learning_rate = 8e-5
        config.actor_adam_epsilon = 1e-8
        config.actor_clip_by_norm = True
        config.actor_clip_norm = 100.0
        config.actor_weight_decay = 0.0

        config.critic_learning_rate = 8e-5
        config.critic_adam_epsilon = 1e-8
        config.critic_clip_by_norm = True
        config.critic_clip_norm = 100.0
        config.critic_weight_decay = 0.0
        
        config.ensemble_learning_rate = 6e-4
        config.ensemble_adam_epsilon = 1e-8
        config.ensemble_clip_by_norm = True
        config.ensemble_clip_norm = 100.0
        config.ensemble_weight_decay = 0.0

        config.imagine_horizon = 15

        config.entropy_bonus = 0.0

        config.add_subconf(name="objective",
                           sub_conf=self._objective_factory.get_default_config(finalize_adding=finalize_adding))

        config.eval_interval = -1

        if finalize_adding:
            config.finalize_adding()
        return config
        
    def build(self,
              policy: LIPolicy,
              model: AbstractSSM,
              ensemble: Optional[EnsembleModel],
              config: ConfigDict) -> UncertainLIPolicyTrainer:

        model_objective = self._objective_factory.build(model=model,
                                                        config=config.objective)

        return UncertainLIPolicyTrainer(policy=policy,
                                        model_objective=model_objective,
                                        ensemble=ensemble,
                                        lambda_=config.lambda_,
                                        discount=config.discount,
                                        imagine_horizon=config.imagine_horizon,
                                        model_learning_rate=config.model_learning_rate,
                                        model_adam_eps=config.model_adam_epsilon,
                                        model_clip_norm=config.model_clip_norm,
                                        model_weight_decay=config.model_weight_decay,
                                        actor_learning_rate=config.actor_learning_rate,
                                        actor_adam_eps=config.actor_adam_epsilon,
                                        actor_clip_norm=config.actor_clip_norm,
                                        actor_weight_decay=config.actor_weight_decay,
                                        critic_learning_rate=config.critic_learning_rate,
                                        critic_adam_eps=config.critic_adam_epsilon,
                                        critic_clip_norm=config.critic_clip_norm,
                                        critic_weight_decay=config.critic_weight_decay,
                                        ensemble_learning_rate=config.ensemble_learning_rate,
                                        ensemble_adam_eps=config.ensemble_adam_epsilon,
                                        ensemble_clip_norm=config.ensemble_clip_norm,
                                        ensemble_weight_decay=config.ensemble_weight_decay,
                                        entropy_bonus=config.entropy_bonus,
                                        eval_interval=config.eval_interval)

