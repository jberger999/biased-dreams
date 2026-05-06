import torch

nn = torch.nn
jit = torch.jit
F = torch.nn.functional


class OneHotCategoricalStraightThrough:
    """
    JIT friendly OneHotCategoricalStraightThrough object that does not use torch.distributions.Categorical,
    since it is not supported.
    """
    @staticmethod
    def mode(probs: torch.Tensor) -> torch.Tensor:
        # probs.shape = [..., num_cat, cat_size]
        
        # Normalize probs
        norm_probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Max prob sample (NOTE: not differentiable!)
        indices = torch.argmax(norm_probs, dim=-1)
        return F.one_hot(indices, probs.shape[-1]).to(probs)

    @staticmethod
    def rsample(probs: torch.Tensor) -> torch.Tensor:
        # probs.shape = [..., num_cat, cat_size]
        
        # Normalize probs
        norm_probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Flatten, get indices, reshape, since multinomial needs len(prob.shape) < 3
        flat_probs = norm_probs.reshape(-1, probs.shape[-1])
        flat_indices = torch.multinomial(flat_probs, 1).squeeze(-1)
        indices = flat_indices.reshape(-1, probs.shape[-2])

        # One-hot sample depending on probs
        samples = F.one_hot(indices, probs.shape[-1]).to(probs)

        # Straight-through estimator
        return samples + (norm_probs - norm_probs.detach())