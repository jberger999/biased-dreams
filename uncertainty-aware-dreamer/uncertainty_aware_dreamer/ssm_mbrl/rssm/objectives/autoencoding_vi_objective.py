import torch
from typing import Optional
from collections import OrderedDict

from uncertainty_aware_dreamer.ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from uncertainty_aware_dreamer.ssm_mbrl.rssm.rssm import RSSM

nn = torch.nn

class AEVIObjective(AbstractObjective):

    def __init__(self,
                 rssm: RSSM,
                 kl_objective: nn.Module,
                 decoder_loss_scales: list[float] = [1.0]):

        super(AEVIObjective, self).__init__(model=rssm,
                                            decoder_loss_scales=decoder_loss_scales)
        self._rssm = rssm
        self._kl_objective = kl_objective
        
    def compute_states(self,
                       observations: list[torch.Tensor],
                       targets: list[torch.Tensor],
                       actions: torch.Tensor,
                       loss_masks: Optional[list[Optional[torch.Tensor]]] = None,
                       return_prior_states: bool = False):
        """
        Computes the posterior or prior states of the RSSM.
        """        
        embedded_obs = self._rssm.encode(observations=observations)
        post_states, prior_states = self._rssm.transition_model.forward_pass(embedded_obs=embedded_obs,
                                                                             actions=actions)
        return prior_states if return_prior_states else post_states
            
    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  targets: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  loss_masks: Optional[list[Optional[torch.Tensor]]] = None):
        
        embedded_obs = self._rssm.encode(observations=observations)
        post_states, prior_states = self._rssm.transition_model.forward_pass(embedded_obs=embedded_obs,
                                                                             actions=actions)
        dec_features = self._rssm.transition_model.get_features(post_states)
        recon_loss, recon_lls, recon_mses = self._get_reconstruction_losses(dec_features=dec_features,
                                                                            targets=targets,
                                                                            element_wise_loss_masks=loss_masks)

        kl_term, kl_dict = self._kl_objective(post_states=post_states, prior_states=prior_states)

        elbo = recon_loss - kl_term
        return - elbo, self._build_log_dict(recon_lls=recon_lls, recon_mses=recon_mses, kl_dict=kl_dict), post_states, embedded_obs