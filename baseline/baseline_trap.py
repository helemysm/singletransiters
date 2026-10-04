import glob
import os
import time

import numpy as np
from scipy.optimize import minimize

from core.config import cfg

from baseline.baseline_ls import _bls_init

DEPTH_MIN_PPM = 300.0
DEPTH_MAX_PPM = 25000#15000.0
DUR_MIN_H     = 1.5
DUR_MAX_H     = 15.0

DEPTH_MIN = DEPTH_MIN_PPM * 1e-6
DEPTH_MAX = DEPTH_MAX_PPM * 1e-6
DUR_MIN_D = DUR_MIN_H / 24.0
DUR_MAX_D = DUR_MAX_H / 24.0

T0_SEARCH_H = 8.0
T0_SEARCH_D = T0_SEARCH_H / 24.0

TAU_FRAC_MIN = 0.01
TAU_FRAC_MAX = 0.50

NM_MAXITER = 200   # Nelder-Mead: max iterations


def trapezoid_model(t, depth, t0, T14, tau):
    
    half = 0.5 * T14
    dt = np.abs(t - t0)
    flat_half = max(half - tau, 0.0) # half-width of the flat bottom

    f = np.zeros_like(t)
    # flat bottom: |dt| <= flat_half  -> -depth
    in_flat = dt <= flat_half
    f[in_flat] = -depth
    # flanks: flat_half < |dt| < half -> linear ramp
    if tau > 0:
        in_ramp = (dt > flat_half) & (dt < half)
        
        frac = (half - dt[in_ramp]) / tau # 1 at flat_half, 0 at half
        f[in_ramp] = -depth * np.clip(frac, 0.0, 1.0)
    return f


def _clip_params(depth, t0, T14, tau_frac, t_lo, t_hi):
    depth    = float(np.clip(depth, DEPTH_MIN, DEPTH_MAX))
    T14      = float(np.clip(T14, DUR_MIN_D, DUR_MAX_D))
    tau_frac = float(np.clip(tau_frac, TAU_FRAC_MIN, TAU_FRAC_MAX))
    t0       = float(np.clip(t0, t_lo, t_hi))
    return depth, t0, T14, tau_frac


def fit_transit(t_days, flux):
    
    t = np.asarray(t_days, dtype=float)
    baseline = np.median(flux)
    f = np.asarray(flux, dtype=float) - baseline   

    win_lo, win_hi = float(t.min()), float(t.max())

    depth0, t0_0, half_dur0 = _bls_init(t, f)
    T14_0 = float(np.clip(2.0 * half_dur0, DUR_MIN_D, DUR_MAX_D))
    depth0 = float(np.clip(depth0, DEPTH_MIN, DEPTH_MAX))
    t0_0   = float(np.clip(t0_0, win_lo, win_hi))
    tau_frac0 = 0.1

    # t0 bounded to a neighborhood of the box-init (intersected with the window).
    # The box already locates the event via the BLS grid; the trapezoid only refines the shape around it, it does not re-search the position -> avoids
    # escaping to the edges.
    t_lo = max(win_lo, t0_0 - T0_SEARCH_D)
    t_hi = min(win_hi, t0_0 + T0_SEARCH_D)

    p0 = np.array([depth0, t0_0, T14_0, tau_frac0], dtype=float)

    def cost(p):
        depth, t0, T14, tau_frac = _clip_params(p[0], p[1], p[2], p[3], t_lo, t_hi)
        tau = tau_frac * T14
        model = trapezoid_model(t, depth, t0, T14, tau)
        resid = f - model
        return float(np.dot(resid, resid))   # SSR

    res = minimize(
        cost, p0, method="Nelder-Mead",
        options={"maxiter": NM_MAXITER, "xatol": 1e-7, "fatol": 1e-12},
    )

    depth, t0, T14, tau_frac = _clip_params(
        res.x[0], res.x[1], res.x[2], res.x[3], t_lo, t_hi
    )

    return (
        max(depth, 0.0), T14,
        t0, 
    )


def _test_indices(n, train_frac=0.80, val_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    return idx[n_train + n_val:]


def predict_trap(npz_path, indices=None):
    
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float64) # (N, L, 2)
    y = d["y"].astype(np.float64)# (N, 3)

    if indices is None:
        indices = np.arange(X.shape[0])

    N = len(indices)
    y_pred = np.empty((N, 3))
    t0_wall = time.time()

    for k, i in enumerate(indices):
        depth_frac, dur_days, dtcorr_days = fit_transit(X[i, :, 0], X[i, :, 1])
        y_pred[k] = [depth_frac * 1e6, dur_days * 24.0, dtcorr_days * 24.0]

        if (k + 1) % 100 == 0 or k == N - 1:
            elapsed = time.time() - t0_wall
            rate   = (k + 1) / elapsed
            avg_ms = elapsed / (k + 1) * 1000
            print(f"  {k+1:>5d}/{N}  ({rate:.1f} samples/s | avg {avg_ms:.1f} ms/LC)",
                  end="\r", flush=True)
    print()

    y_sub  = y[indices]
    y_true = np.column_stack([y_sub[:, 0] * 1e6, y_sub[:, 1] * 24.0, y_sub[:, 2] * 24.0])
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    err  = y_pred - y_true
    return {
        "N": len(y_true),
        "mae":  np.mean(np.abs(err), axis=0),
        "rmse": np.sqrt(np.mean(err ** 2, axis=0)),
        "bias": np.mean(err, axis=0),
    }


def pretty_print(tag, m):
    names = ["depth_ppm", "duration_h", "dtcorr_h"]
    print(f"== {tag} (trapezoid baseline) ==")
    print(f"N = {m['N']}")
    for j, name in enumerate(names):
        print(f"{name:>10s} | MAE={m['mae'][j]:9.3f} | RMSE={m['rmse'][j]:9.3f}"
              f" | bias={m['bias'][j]:+.3f}")


def main():
    EVAL_MODE = "data_training"   # "data_training" or "sensitivity"
    NPZ_PATH  = cfg["paths"]["train_data"]
    SENS_DIR  = cfg["paths"]["sensitivity_sets"]

    if EVAL_MODE == "data_training":
        print(f"Evaluating trapezoid on test split of: {NPZ_PATH}")
        n = int(np.load(NPZ_PATH, allow_pickle=True)["X"].shape[0])
        test_idx = _test_indices(n, seed=42)
        print(f"Test set size: {len(test_idx)}")
        y_true, y_pred = predict_trap(NPZ_PATH, indices=test_idx)
        pretty_print("TEST", compute_metrics(y_true, y_pred))

    elif EVAL_MODE == "sensitivity":
        npz_files = sorted(glob.glob(os.path.join(SENS_DIR, "sens_*.npz")))
        if not npz_files:
            print(f"No sensitivity files in '{SENS_DIR}'.")
            return
        for path in npz_files:
            tag = os.path.splitext(os.path.basename(path))[0]
            print(f"--- {tag} ---")
            y_true, y_pred = predict_trap(path)
            pretty_print(tag, compute_metrics(y_true, y_pred))


if __name__ == "__main__":
    main()