import torch

jit = torch.jit


class AbstractDistanceMeasure(jit.ScriptModule):
    
    def __init__(self):
        super(AbstractDistanceMeasure, self).__init__()

 
    @jit.script_method
    def compute_measure(self,
                        means: torch.Tensor,
                        vars: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance measure between ensemble predictions.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class GeometricJensenShannonDivergence(AbstractDistanceMeasure):
    
    def __init__(self):
        super(GeometricJensenShannonDivergence, self).__init__()


    @jit.script_method
    def compute_measure(self,
                        means: torch.Tensor,
                        vars: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean GJS-divergence between ensemble predictions to quantify epistemic / model uncertainty.
        Output shape: (batch_size, batch_length, 1)
        """ 
        num_gaussians = means.shape[0]
        
        # Compute pairwise GJS-divergences.
        gjs_divergences = jit.annotate(list[torch.Tensor], [])
        for i in range(num_gaussians):
            for j in range(i):
                gjs_div = self._geometric_jensen_shannon_divergence(means_first_gaussian=means[i],
                                                                    vars_first_gaussian=vars[i],
                                                                    means_second_gaussian=means[j],
                                                                    vars_second_gaussian=vars[j])               
                gjs_divergences.append(gjs_div)
        
        mean_gjs_divergence = torch.stack(gjs_divergences, -1).sum(-1, keepdim=True) / len(gjs_divergences)
        
        assert not mean_gjs_divergence.isnan().any(), "Mean GJS-divergences contain NaNs."
        
        return mean_gjs_divergence


    @jit.script_method
    def _geometric_jensen_shannon_divergence(self,
                                             means_first_gaussian: torch.Tensor,
                                             vars_first_gaussian: torch.Tensor,
                                             means_second_gaussian: torch.Tensor,
                                             vars_second_gaussian: torch.Tensor) -> torch.Tensor:
        """
        Computes the closed-form solution of the GJS-divergence between two Gaussian distributions with a diagonal covariance matrix each.
        
        Original code from MACURA: https://github.com/Data-Science-in-Mechanical-Engineering/macura/blob/master/mbrl/util/distance_measures.py
        """
        al = 0.5
        t1 = (1 - al) * means_first_gaussian / vars_first_gaussian * means_first_gaussian
        t1S = torch.sum(t1, -1)
        t2 = al * means_second_gaussian / vars_second_gaussian * means_second_gaussian
        t2S = torch.sum(t2, -1)
        sigAL = 1 / ((1 - al) / vars_first_gaussian + al / vars_second_gaussian)
        muAl = sigAL * ((1 - al) / vars_first_gaussian * means_first_gaussian + al / vars_second_gaussian * means_second_gaussian)
        t3 = muAl / sigAL * muAl
        t3S = torch.sum(t3, -1)
        log_det_S1 = torch.sum(torch.log(vars_first_gaussian), -1)
        log_det_S2 = torch.sum(torch.log(vars_second_gaussian), -1)
        log_det_SSum = torch.sum(torch.log(sigAL), -1)
        log_term = (1 - al) * log_det_S1 + al * log_det_S2 - log_det_SSum
        shanon = 1 / 2 * (t1S + t2S - t3S + log_term)
        return shanon


class JensenRenyiDivergence(AbstractDistanceMeasure):
    
    def __init__(self):
        super(JensenRenyiDivergence, self).__init__()
    

    @jit.script_method
    def compute_measure(self,
                        means: torch.Tensor,
                        vars: torch.Tensor) -> torch.Tensor:
        
        """
        Compute the Jensen-Renyi divergence between ensemble predictions to quantify epistemic / model uncertainty.
        Output shape: (batch_size, batch_length, 1)
        """
        return self._jensen_renyi_divergence(states_mean=means,
                                             states_var=vars,
                                             chunk_size=1000)
    

    @jit.script_method
    def _jensen_renyi_divergence(self,
                                 states_mean: torch.Tensor,
                                 states_var: torch.Tensor,
                                 chunk_size: int = 1000) -> torch.Tensor:
        """
        Memory-efficient Jensen-Renyi divergence computation for a set of states by chunking over actors (n_act).
        Computes the Jensen-Renyi divergence for a set of states.
        Adapted from: https://github.com/CMU-IntentLab/UNISafe/blob/isaaclab/dreamerv3_torch/ensemble/utility.py

        Args:
            states_mean (torch.Tensor): Mean of the states, shape (ensemble_size, n_actors, d_state).
            states_var (torch.Tensor): Variance of the states, shape (ensemble_size, n_actors, d_state).

        Returns:
            torch.Tensor: The computed Jensen-Renyi divergence, shape (n_actors,).
        """
        if len(states_mean.shape) == 4:
            _, B, N, _ = states_mean.shape                  # [es, B, N, d_s]
            mu = states_mean.flatten(1, 2).permute(1, 0, 2) # [n_act, es, d_s]
            var = states_var.flatten(1, 2).permute(1, 0, 2) # [n_act, es, d_s]
        elif len(states_mean.shape) == 3:
            _, B, _ = states_mean.shape        # [es, B, d_s]
            N = -1
            mu = states_mean.permute(1, 0, 2)  # [n_act, es, d_s]
            var = states_var.permute(1, 0, 2)  # [n_act, es, d_s]
        else:
            B, N = -1, -1
            mu = states_mean.unsqueeze(1)
            var = states_var.unsqueeze(1)

        n_act, es, d_s = mu.size()
        
        entropy_mean_chunks = []
        for start in range(0, n_act, chunk_size):
            end = min(start + chunk_size, n_act)
            
            mu_chunk = mu[start:end]    # [chunk_size, es, d_s]
            var_chunk = var[start:end]  # [chunk_size, es, d_s]

            # ----- Entropy of Mean -----
            # Delta = mu_i - mu_j, Phi = var_i + var_j
            mu_diff = mu_chunk.unsqueeze(2) - mu_chunk.unsqueeze(1)     # [chunk, es, es, d_s]
            var_sum = var_chunk.unsqueeze(2) + var_chunk.unsqueeze(1)   # [chunk, es, es, d_s]

            err = (mu_diff * (1 / var_sum) * mu_diff).sum(dim=-1)       # [chunk, es, es]
            det = torch.log(var_sum).sum(dim=-1)                        # [chunk, es, es]

            log_z = -0.5 * (err + det)                                  # [chunk, es, es]
            log_z = log_z.view(end - start, es * es)                    # [chunk, es*es]

            mx, _ = log_z.max(dim=1, keepdim=True)                      # [chunk, 1]
            log_z = log_z - mx
            exp = torch.exp(log_z).mean(dim=1, keepdim=True)            # [chunk, 1]

            entropy_mean_chunk = -mx - torch.log(exp)                   # [chunk, 1]
            entropy_mean_chunks.append(entropy_mean_chunk.squeeze(1))   # [chunk]

        entropy_mean = torch.cat(entropy_mean_chunks, dim=0)            # [n_act]

        # ----- Mean of Entropies -----
        total_entropy = torch.sum(torch.log(var), dim=-1)               # [n_act, es]
        mean_entropy = total_entropy.mean(dim=1) / 2 + d_s * 0.5 * torch.log(torch.tensor(2.0))
        
        utility = entropy_mean - mean_entropy  # [n_act]
        
        # Reshape to original dimensions
        if len(states_mean.shape) == 4:
            utility = utility.view(B, N, 1)
        elif len(states_mean.shape) == 3:
            utility = utility.view(B, 1)
        else:
            utility = utility.view(-1, 1)

        return utility