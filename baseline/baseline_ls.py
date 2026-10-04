import glob
import os
import time

import numpy as np
from core.config import cfg

EVAL_MODE = "sensitivity"   # "data_training" | "sensitivity"
NPZ_PATH  = cfg["paths"]["train_data"]
SENS_DIR  = cfg["paths"]["sensitivity_sets"]
SPLIT_SEED = 42             # must match train.py / data.py


def _bls_init(t, f):
    
    DEPTH_MIN_PPM = 300.0
    DEPTH_MAX_PPM = 15000.0
    DUR_MIN_H     = 1.5
    DUR_MAX_H     = 15.0
    
    L = len(t)
    cumf = np.empty(L + 1)
    cumf[0] = 0.0
    np.cumsum(f, out=cumf[1:])

    cadence_days = float(np.median(np.diff(t))) if L > 1 else (t[-1] - t[0]) / (L - 1)
    n_min = max(3, int(DUR_MIN_H / 24.0 / cadence_days))
    n_max = int(DUR_MAX_H / 24.0 / cadence_days)

    # Upper-triangle indices: i2 >= i1 + 3  (i2 indexes cumf → n_in = i2 - i1)
    i1, i2 = np.triu_indices(L + 1, k=3)
    n_in = i2 - i1
    keep  = (n_in >= n_min) & (n_in <= n_max) # enforce duration constraints
    i1, i2, n_in = i1[keep], i2[keep], n_in[keep]

    n_out    = L - n_in
    s_in     = cumf[i2] - cumf[i1]
    mean_in  = s_in / n_in
    mean_out = (cumf[L] - s_in) / n_out
    depth    = mean_out - mean_in  

    power = np.where(depth > 0, depth * depth * n_in * n_out, -np.inf)
    kb    = int(np.argmax(power))

    i1b, i2b  = int(i1[kb]), int(i2[kb])
    i2b_clamp = min(i2b - 1, L - 1)
    t0_init       = 0.5 * (t[i1b] + t[i2b_clamp])
    half_dur_init = 0.5 * max(t[i2b_clamp] - t[i1b], 1e-5)
    depth_init    = max(float(depth[kb]), 1e-6)

    return depth_init, t0_init, half_dur_init


def fit_transit(t_days, flux):
    
    baseline = np.median(flux)
    f = flux - baseline 

    depth0, t0_0, hd0 = _bls_init(t_days, f)

    return (
        max(depth0, 0.0),
        max(2.0 * hd0, 1e-4),
        t0_0,
    )


def _test_indices(n, train_frac=0.80, val_frac=0.10, seed=42):
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    return idx[n_train + n_val:]


def predict_ls(npz_path, indices=None):
    
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float64) # (N, L, 2): col 0 = t_rel_days, col 1 = flux
    y = d["y"].astype(np.float64) # (N, 3):   [depth_frac, dur_days, dtcorr_days]

    if indices is None:
        indices = np.arange(X.shape[0])

    N      = len(indices)
    y_pred = np.empty((N, 3))
    t0_wall = time.time()

    for k, i in enumerate(indices):
        depth_frac, dur_days, dtcorr_days = fit_transit(X[i, :, 0], X[i, :, 1])
        y_pred[k] = [depth_frac * 1e6, dur_days * 24.0, dtcorr_days * 24.0]

        if (k + 1) % 100 == 0 or k == N - 1:
            elapsed = time.time() - t0_wall
            rate    = (k + 1) / elapsed
            avg_ms  = elapsed / (k + 1) * 1000
            print(f"  {k+1:>5d}/{N}  ({rate:.1f} samples/s | avg per LC: {avg_ms:.1f} ms)", end="\r", flush=True)

    print()

    y_sub  = y[indices]
    y_true = np.column_stack([
        y_sub[:, 0] * 1e6,
        y_sub[:, 1] * 24.0,
        y_sub[:, 2] * 24.0,
    ])
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    err  = y_pred - y_true
    mae  = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    bias = np.mean(err, axis=0)
    return {"N": len(y_true), "mae": mae, "rmse": rmse, "bias": bias}


def pretty_print_ls(tag, m):
    names = ["depth_ppm", "duration_h",  "dtcorr_h"]
    print(f"\n== {tag} (LS baseline) ==")
    print(f"N = {m['N']}")
    for j, name in enumerate(names):
        print(
            f"{name:>10s} | MAE={m['mae'][j]:9.3f} | RMSE={m['rmse'][j]:9.3f}"
            f" | bias={m['bias'][j]:+.3f}"
        )


def main():
    if EVAL_MODE == "data_training":
        print(f"Evaluating on test split of: {NPZ_PATH}")
        n = int(np.load(NPZ_PATH, allow_pickle=True)["X"].shape[0])
        test_idx = _test_indices(n, seed=SPLIT_SEED)
        print(f"Test set size: {len(test_idx)}")
        y_true, y_pred = predict_ls(NPZ_PATH, indices=test_idx)
        pretty_print_ls("TEST", compute_metrics(y_true, y_pred))

    elif EVAL_MODE == "sensitivity":
        npz_files = sorted(glob.glob(os.path.join(SENS_DIR, "sens_*.npz")))
        if not npz_files:
            print(f"No sensitivity files found in '{SENS_DIR}'.")
            return
        print(f"Found {len(npz_files)} sensitivity files.\n")

        for path in npz_files:
            tag = os.path.splitext(os.path.basename(path))[0]
            print(f"--- {tag} ---")
            y_true, y_pred = predict_ls(path)
            pretty_print_ls(tag, compute_metrics(y_true, y_pred))

    else:
        raise ValueError(f"Unknown EVAL_MODE='{EVAL_MODE}'")


if __name__ == "__main__":
    main()