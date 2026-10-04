import math
import torch


#Heteroscedastic Gaussian NLL
def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * ((y - mu) ** 2) * inv_var + log_sigma + 0.5 * math.log(2.0 * math.pi)
    return nll.mean()

def gaussian_nll_weighted(y, mu, log_sigma, weights=None):
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * ((y - mu) ** 2) * inv_var + log_sigma + 0.5 * math.log(2.0 * math.pi)
    if weights is not None:
        nll = nll * weights
    return nll.mean()

def beta_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    beta: float = 0.5,
    beta_per_dim: torch.Tensor | None = None,
) -> torch.Tensor:
    
    #beta-NLL (Seitzer et al., 2022)
    # 
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * ((y - mu) ** 2) * inv_var + log_sigma + 0.5 * math.log(2.0 * math.pi)

    if beta_per_dim is not None:
        b = beta_per_dim.to(log_sigma.device).view(1, -1)   # (1, D)
    else:
        b = beta

    weight = torch.exp(2.0 * log_sigma * b).detach()         # (sigma^2)^beta
    return (weight * nll).mean()