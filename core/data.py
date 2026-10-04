import numpy as np
import torch
from torch.utils.data import Dataset


def make_splits(
    n: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


class TransitNPZDataset(Dataset):
    """
    Loads:
      X: (N, L, 2) channels=[t_rel_days, flux]
      y: (N, 3) labels=[depth_frac, duration_days, dtcorr_days]

    Produces:
      x: (C, L) float32, C=2 if use_time else 1
      y: (3,) float32 in units [depth_ppm, duration_h, dtcorr_h]
    """

    def __init__(self, npz_path: str, use_time: bool = True, center_flux: bool = True):
        d = np.load(npz_path, allow_pickle=True)
        X = d["X"].astype(np.float32)  # (N, L, 2)
        y_raw = d["y"].astype(np.float32)  # (N, 3)

        self.use_time = use_time
        self.center_flux = center_flux

        # time
        t_days = X[:, :, 0]
        t_h = t_days * 24.0
        half_range = 0.5 * (np.max(t_h, axis=1) - np.min(t_h, axis=1))
        half_range = np.maximum(half_range, 1e-6)
        self.t_scaled = (t_h / half_range[:, None]).astype(np.float32)  # (N, L)

        # flux channel
        self.flux = X[:, :, 1].astype(np.float32)
        if self.center_flux:
            self.flux = self.flux - 1.0

        # targets
        depth_ppm = y_raw[:, 0] * 1e6
        duration_h = y_raw[:, 1] * 24.0
        dtcorr_h = y_raw[:, 2] * 24.0
        self.y = np.stack([depth_ppm, duration_h, dtcorr_h], axis=1).astype(np.float32)

    def __len__(self):
        return self.flux.shape[0]

    def __getitem__(self, idx):
        if self.use_time:
            x = np.stack([self.t_scaled[idx], self.flux[idx]], axis=0)  # (2, L)
        else:
            x = self.flux[idx][None, :]  # (1, L)
        y = self.y[idx]
        return torch.from_numpy(x), torch.from_numpy(y)