import torch
from collections import OrderedDict

from uncertainty_aware_dreamer.ssm_mbrl.mbrl.common.abstract_trainer import AbstractTrainer
from uncertainty_aware_dreamer.ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from uncertainty_aware_dreamer.ssm_mbrl.mbrl.latent_imagination.latent_imagination_policy import LIPolicy
from uncertainty_aware_dreamer.ssm_mbrl.util.freeze_parameters import FreezeParameters
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.ensemble import EnsembleModel

nn = torch.nn
opt = torch.optim
data = torch.utils.data


class LIPolicyTrainer(AbstractTrainer):

    def __init__(self,
                 policy: LIPolicy,
                 model_objective: AbstractObjective,
                 discount: float,
                 lambda_: float,
                 imagine_horizon: int,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float,
                 actor_learning_rate: float,
                 actor_adam_eps: float,
                 actor_clip_norm: float,
                 actor_weight_decay: float,
                 critic_learning_rate: float,
                 critic_adam_eps: float,
                 critic_clip_norm: float,
                 critic_weight_decay: float,
                 entropy_bonus: float = 0.0,
                 eval_interval: int = 1):
        super().__init__(objective=model_objective,
                         model=policy.model,
                         eval_interval=eval_interval)

        self._policy = policy

        self._imagine_horizon = imagine_horizon
        self._discount = discount
        self._lambda = lambda_

        self._entropy_bonus = entropy_bonus

        self._model_optimizer, self._model_clip_fn = \
            self._build_optimizer_and_clipping(params=self._objective.parameters(),
                                               learning_rate=model_learning_rate,
                                               adam_eps=model_adam_eps,
                                               clip_norm=model_clip_norm,
                                               weight_decay=model_weight_decay)
        self._actor_optimizer, self._actor_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.actor.parameters(),
                                               learning_rate=actor_learning_rate,
                                               adam_eps=actor_adam_eps,
                                               clip_norm=actor_clip_norm,
                                               weight_decay=actor_weight_decay)
        self._critic_optimizer, self._critic_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.critic.parameters(),
                                               learning_rate=critic_learning_rate,
                                               adam_eps=critic_adam_eps,
                                               clip_norm=critic_clip_norm,
                                               weight_decay=critic_weight_decay)

    def get_optimizer_state_dict(self) -> dict:
        return {"model": self._model_optimizer.state_dict(),
                "actor": self._actor_optimizer.state_dict(),
                "critic": self._critic_optimizer.state_dict()}

    def load_optimizer_state_dict(self, state_dict: dict):
        self._model_optimizer.load_state_dict(state_dict=state_dict["model"])
        self._actor_optimizer.load_state_dict(state_dict=state_dict["actor"])
        self._critic_optimizer.load_state_dict(state_dict=state_dict["critic"])

    def _train_on_batch(self, batch):
        # --------- MODEL LOSS --------- #
        model_loss, model_obj_log_dict, post_states, _ = self._objective.compute_losses_and_states(*batch)
            
        # --------- ACTOR LOSS --------- #
        with FreezeParameters([self._model, self._policy.critic]):
            # Combine the first two dimensions and detach initial states.
            size = post_states["sample"].shape[0] * post_states["sample"].shape[1]
            initial_states = {k: v.detach().reshape(size, *v.shape[2:]) for k, v in post_states.items()}
            # Rollout policy for a certain amount of steps without incorporating observations.
            imagined_states, action_log_probs, _ = \
                self._model.rollout_policy(state=initial_states,
                                        policy_fn=self._policy.actor.get_sampled_action_and_log_prob,
                                        num_steps=self._imagine_horizon)
            # Extract samples from imagined states.
            imagined_features = self._model.get_features(state=imagined_states)
            # Predict rewards and values from imagined features.
            rewards = self._model.decoders[1](imagined_features)[0]
            values = self._policy.critic(imagined_features)

            # Compute lambda returns from predicted rewards and values.
            lambda_returns = self.compute_generalized_values(rewards=rewards[:, :-1],
                                                             values=values[:, :-1],
                                                             bootstrap=values[:, -1],
                                                             discount=self._discount,
                                                             lambda_=self._lambda)
            actor_entropy = - action_log_probs.mean()
            scaled_actor_entropy = self._entropy_bonus * actor_entropy
            # The actors objective is to maximize the expected return.
            actor_loss = - (lambda_returns.mean() + scaled_actor_entropy)

        # -------- CRITIC LOSS --------- #
        with FreezeParameters([self._model, self._policy.actor]):
            with torch.no_grad():
                # :-1 to match the length of the lambda returns.
                detached_imagined_features = imagined_features[:, :-1].detach()
                detached_target_values = lambda_returns.detach()
            # The critic objective is to match its predicted values to the lambda returns.
            out_values = self._policy.critic(detached_imagined_features)
            critic_loss = 0.5 * (detached_target_values - out_values).square().mean()
        
        self._model_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()

        model_loss.backward()
        actor_loss.backward()
        critic_loss.backward()

        self._model_clip_fn(self._model.parameters())
        self._actor_clip_fn(self._policy.actor.parameters())
        self._critic_clip_fn(self._policy.critic.parameters())

        self._model_optimizer.step()
        self._actor_optimizer.step()
        self._critic_optimizer.step()

        # --------- LOGGING --------- #
        log_dict = OrderedDict({"model/neg_elbo": model_loss.detach_().cpu().numpy()},
                               **{"model/{}".format(k): v for k, v in model_obj_log_dict.items()})
        log_dict["actor/loss"] = actor_loss.detach_().cpu().numpy()
        log_dict["actor/loss"] = critic_loss.detach_().cpu().numpy()
        log_dict["actor/entropy"] = actor_entropy.detach_().cpu().numpy()

        return log_dict

    @staticmethod
    def compute_generalized_values(rewards: torch.Tensor,
                                   values: torch.Tensor,
                                   bootstrap: torch.Tensor,
                                   discount: float = 0.99,
                                   lambda_: float = 0.95) -> torch.Tensor:
        next_values = torch.cat([values[:, 1:], bootstrap[:, None]], 1)
        deltas = rewards + discount * next_values * (1 - lambda_)
        last = bootstrap
        returns = torch.ones_like(rewards)
        for t in reversed(range(rewards.shape[1])):
            returns[:, t] = last = deltas[:, t] + (discount * lambda_ * last)
        return returns
    
    
class UncertainLIPolicyTrainer(LIPolicyTrainer):
    
    def __init__(self,
                 policy: LIPolicy,
                 model_objective: AbstractObjective,
                 ensemble: EnsembleModel,
                 discount: float,
                 lambda_: float,
                 imagine_horizon: int,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float,
                 actor_learning_rate: float,
                 actor_adam_eps: float,
                 actor_clip_norm: float,
                 actor_weight_decay: float,
                 critic_learning_rate: float,
                 critic_adam_eps: float,
                 critic_clip_norm: float,
                 critic_weight_decay: float,
                 ensemble_learning_rate: float,
                 ensemble_adam_eps: float,
                 ensemble_clip_norm: float,
                 ensemble_weight_decay: float,
                 entropy_bonus: float = 0.0,
                 eval_interval: int = 1):
        super().__init__(policy=policy,
                         model_objective=model_objective,
                         discount=discount,
                         lambda_=lambda_,
                         imagine_horizon=imagine_horizon,
                         model_learning_rate=model_learning_rate,
                         model_adam_eps=model_adam_eps,
                         model_clip_norm=model_clip_norm,
                         model_weight_decay=model_weight_decay,
                         actor_learning_rate=actor_learning_rate,
                         actor_adam_eps=actor_adam_eps,
                         actor_clip_norm=actor_clip_norm,
                         actor_weight_decay=actor_weight_decay,
                         critic_learning_rate=critic_learning_rate,
                         critic_adam_eps=critic_adam_eps,
                         critic_clip_norm=critic_clip_norm,
                         critic_weight_decay=critic_weight_decay,
                         entropy_bonus=entropy_bonus,
                         eval_interval=eval_interval)
        
        self._ensemble = ensemble
        
        # Build optimizer for ensemble, if available.
        if self._ensemble is not None:
            self._ensemble_optimizer, self._ensemble_clip_fn = \
                self._build_optimizer_and_clipping(params=self._ensemble.parameters(),
                                                   learning_rate=ensemble_learning_rate,
                                                   adam_eps=ensemble_adam_eps,
                                                   clip_norm=ensemble_clip_norm,
                                                   weight_decay=ensemble_weight_decay)
        else:
            self._ensemble_optimizer, self._ensemble_clip_fn = None, None
            
        
    def _train_on_batch(self, batch):
        # --------- MODEL LOSS --------- #
        # Includes (separate) loss of ensemble for URSSM version.
        model_loss, model_obj_log_dict, post_states, embedded_obs = self._objective.compute_losses_and_states(*batch)
        # -------- ENSEMBLE LOSS ------- #
        with FreezeParameters([self._model, self._policy.actor, self._policy.critic]):
            if self._ensemble is not None:
                features = self._model.get_features(state=post_states)[:, :-1].detach()
                actions = batch[2][:, 1:]
                targets = self._ensemble.get_ensemble_target(self._model, post_states, embedded_obs)[:, 1:].detach()
                ensemble_loss, ensemble_log_dict = self._ensemble.compute_loss(states=features,
                                                                               actions=actions,
                                                                               targets=targets)

        # --------- ACTOR LOSS --------- #
        with FreezeParameters([self._model, self._policy.critic]):
            # Combine the first two dimensions and detach initial states.
            size = post_states["sample"].shape[0] * post_states["sample"].shape[1]
            initial_states = {k: v.detach().reshape(size, *v.shape[2:]) for k, v in post_states.items()}
            # Rollout policy for a certain amount of steps without incorporating observations.
            imagined_states, action_log_probs, _ = \
                self._model.rollout_policy(state=initial_states,
                                        policy_fn=self._policy.actor.get_sampled_action_and_log_prob,
                                        num_steps=self._imagine_horizon)
            # Extract samples from imagined states.
            imagined_features = self._model.get_features(state=imagined_states)
            # Predict rewards and values from imagined features.
            rewards = self._model.decoders[1](imagined_features)[0]
            values = self._policy.critic(imagined_features)

            # Compute lambda returns from predicted rewards and values.
            lambda_returns = self.compute_generalized_values(rewards=rewards[:, :-1],
                                                             values=values[:, :-1],
                                                             bootstrap=values[:, -1],
                                                             discount=self._discount,
                                                             lambda_=self._lambda)
            actor_entropy = - action_log_probs.mean()
            scaled_actor_entropy = self._entropy_bonus * actor_entropy
            # The actors objective is to maximize the expected return.
            actor_loss = - (lambda_returns.mean() + scaled_actor_entropy)

        # -------- CRITIC LOSS --------- #
        with FreezeParameters([self._model, self._policy.actor]):
            with torch.no_grad():
                # :-1 to match the length of the lambda returns.
                detached_imagined_features = imagined_features[:, :-1].detach()
                detached_target_values = lambda_returns.detach()
            # The critic objective is to match its predicted values to the lambda returns.
            out_values = self._policy.critic(detached_imagined_features)
            critic_loss = 0.5 * (detached_target_values - out_values).square().mean()
                
        # --------- BACKPROP AND UPDATE --------- #
        self._model_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        if self._ensemble is not None:
            self._ensemble_optimizer.zero_grad()
        
        model_loss.backward()
        actor_loss.backward()
        critic_loss.backward()
        if self._ensemble is not None:
            ensemble_loss.backward()
        
        self._model_clip_fn(self._model.parameters())
        self._actor_clip_fn(self._policy.actor.parameters())
        self._critic_clip_fn(self._policy.critic.parameters())
        if self._ensemble is not None:
            self._ensemble_clip_fn(self._ensemble.parameters())
            
        self._model_optimizer.step()
        self._actor_optimizer.step()
        self._critic_optimizer.step()
        if self._ensemble is not None:
            self._ensemble_optimizer.step()

        # --------- LOGGING --------- #
        log_dict = OrderedDict({"model/neg_elbo": model_loss.detach_().cpu().numpy()},
                               **{"model/{}".format(k): v for k, v in model_obj_log_dict.items()})
        if self._ensemble is not None:
            log_dict |= OrderedDict({**{"ensemble/{}".format(k): v for k, v in ensemble_log_dict.items()}})
        
        log_dict["actor/loss"] = actor_loss.detach_().cpu().numpy()
        log_dict["critic/loss"] = critic_loss.detach_().cpu().numpy()
        log_dict["actor/entropy"] = actor_entropy.detach_().cpu().numpy()
        
        return log_dict
    
    
    @torch.no_grad()
    def compute_uncertainty(self, 
                            imagined_states,
                            actions,
                            align=True):
        
        imagined_features = self._model.get_features(state=imagined_states)
        dis_feat = imagined_features[:, :-1] if align else imagined_features
        dis_act = actions[:, 1:] if align else actions
        uncertainties = self._ensemble.compute_disagreement(state=dis_feat,
                                                            action=dis_act)
        return uncertainties