import numpy as np
import torch
from torch.utils.data import Subset


class TargetNormalizer_old:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps_std: float = 1e-8):
        self.mean = mean.astype(np.float32)
        self.std = (std.astype(np.float32) + eps_std)

    @staticmethod
    def fit_from_subset(ds, subset: Subset):
        ys = []
        for i in subset.indices:
            _, y = ds[i]
            ys.append(y.numpy())
        ys = np.stack(ys, axis=0)
        return TargetNormalizer_old(mean=ys.mean(axis=0), std=ys.std(axis=0))

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.mean).to(y.device)
        std = torch.from_numpy(self.std).to(y.device)
        return (y - mean) / std

    def decode_mu_sigma(self, mu_n: torch.Tensor, log_sigma_n: torch.Tensor):
        """
        If y_n = (y - mean)/std:
          mu = mu_n*std + mean
          sigma = exp(log_sigma_n)*std
        """
        mean = torch.from_numpy(self.mean).to(mu_n.device)
        std = torch.from_numpy(self.std).to(mu_n.device)
        mu = mu_n * std + mean
        sigma = torch.exp(log_sigma_n) * std
        return mu, sigma


#Standardizes the targets (depth, duration, dtcorr).
class TargetNormalizer:
    
    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        log_mask: np.ndarray | None = None,
        eps_std: float = 1e-8,
    ):
        self.mean = mean.astype(np.float32)
        self.std = (std.astype(np.float32) + eps_std)
        if log_mask is None:
            log_mask = np.zeros_like(self.mean, dtype=bool)
        self.log_mask = log_mask.astype(bool)
 
    def _to_internal(self, y: torch.Tensor) -> torch.Tensor:
        
        mask = torch.from_numpy(self.log_mask).to(y.device)
        # log only where mask=True; the rest is unchanged
        return torch.where(mask, torch.log(y), y)
 
    # fit
    @staticmethod
    def fit_from_subset(ds, subset: Subset, log_mask: np.ndarray | None = None):
        ys = []
        for i in subset.indices:
            _, y = ds[i]
            ys.append(y.numpy())
        ys = np.stack(ys, axis=0)  # (N, 3)
 
        # default: log only on duration (col 1)
        if log_mask is None:
            log_mask = np.array([False, True, False], dtype=bool)
        log_mask = log_mask.astype(bool)
 
        # move to internal space before computing mean/std
        ys_internal = ys.copy()
        if log_mask.any():
            cols = np.where(log_mask)[0]
            # guard against non-positive values, just in case
            if (ys[:, cols] <= 0).any():
                raise ValueError(
                    "log_mask flags columns with values <= 0; "
                    "log is undefined. Check that the column is positive."
                )
            ys_internal[:, cols] = np.log(ys[:, cols])
 
        return TargetNormalizer(
            mean=ys_internal.mean(axis=0),
            std=ys_internal.std(axis=0),
            log_mask=log_mask,
        )
 
    # encode / decode
    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """y (physical space) -> y_n (normalized space)."""
        mean = torch.from_numpy(self.mean).to(y.device)
        std = torch.from_numpy(self.std).to(y.device)
        y_int = self._to_internal(y)
        return (y_int - mean) / std
 
    def decode_mu_sigma(self, mu_n: torch.Tensor, log_sigma_n: torch.Tensor):
        
        mean = torch.from_numpy(self.mean).to(mu_n.device)
        std = torch.from_numpy(self.std).to(mu_n.device)
        mask = torch.from_numpy(self.log_mask).to(mu_n.device)
 
        # internal space (log for flagged columns)
        mu_int = mu_n * std + mean
        sigma_int = torch.exp(log_sigma_n) * std
 
        # linear columns: as is
        mu_lin = mu_int
        sigma_lin = sigma_int
 
        # log columns: exponentiate + delta method
        mu_exp = torch.exp(mu_int)
        sigma_exp = mu_exp * sigma_int  # delta method, 1st order
 
        mu = torch.where(mask, mu_exp, mu_lin)
        sigma = torch.where(mask, sigma_exp, sigma_lin)
        return mu, sigma
 