import glob
import os
import csv
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.config import cfg, resolve_log_mask
from core.data import TransitNPZDataset, make_splits
from core.normalizer import TargetNormalizer
from core.calibrate import SigmaCalibrator
from core.metrics import eval_metrics, pretty_print_metrics


EVAL_MODE = "sensitivity"

NPZ_PATH = cfg["paths"]["train_data"]      # used when data_training
SENS_DIR = cfg["paths"]["sensitivity_sets"] # used when sensitivity

MODEL_PATH = cfg["paths"]["model"]
AUX_PATH   = cfg["paths"]["aux"]

OUT_CSV = "eval_summary.csv"

SEED = 42223
BATCH_SIZE = 512

USE_TIME_CHANNEL = True
CENTER_FLUX = True

DUR_BINS = [(0, 4), (4, 8), (8, 12), (12, 16)]

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def _load_model_and_aux():
    aux = np.load(AUX_PATH, allow_pickle=True)
    normalizer = TargetNormalizer(
        mean=aux["y_mean"], std=aux["y_std"], log_mask=resolve_log_mask(aux)
    )
    calib = SigmaCalibrator(d_out=3)
    calib.log_scale.data = torch.tensor(aux["calib_log_scale"], dtype=torch.float32)
    calib = calib.to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    in_ch = 2 if ckpt["use_time_channel"] else 1
    model = cfg["model_class"](
        in_ch=in_ch, d_out=3,
        log_sigma_min=ckpt.get("log_sigma_min", -8.0),
        log_sigma_max=ckpt.get("log_sigma_max",  4.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    return model, normalizer, calib


def _predict(loader, model, normalizer, calib):
    y_true_all, mu_all, sig_all = [], [], []
    total_samples = 0
    t0_wall = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mu_n, log_sigma_n = model(x)
            log_sigma_n = calib(log_sigma_n)
            mu, sigma = normalizer.decode_mu_sigma(mu_n, log_sigma_n)
            y_true_all.append(y.cpu().numpy())
            mu_all.append(mu.cpu().numpy())
            sig_all.append(sigma.cpu().numpy())
            total_samples += x.shape[0]
            elapsed = time.time() - t0_wall
            print(f"  {total_samples} samples  "
                  f"({total_samples/elapsed:.1f} samples/s)", end="\r", flush=True)
    print()
    return np.vstack(y_true_all), np.vstack(mu_all), np.vstack(sig_all)


def _summarize(npz_path, y_true, mu, sigma):
    
    """Compute metrics + duration-bias diagnostics."""
    
    names = ["depth_ppm", "duration_h", "dtcorr_h"]
    err = mu - y_true

    mae  = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    bias = np.mean(err, axis=0)
    med_sigma = np.median(sigma, axis=0)

    # coverage (gaussian): |z| <= 1 and <= 1.96
    z = np.abs(err) / (sigma + 1e-12)
    cov68 = np.mean(z <= 1.0, axis=0)
    cov95 = np.mean(z <= 1.96, axis=0)

    fname = os.path.basename(npz_path)
    print(f"\nSummary — {fname}:")
    for j, name in enumerate(names):
        print(f"  {name:>10s}: MAE={mae[j]:.3f}  RMSE={rmse[j]:.3f}  "
              f"bias={bias[j]:+.3f}  med_sigma={med_sigma[j]:.3f}  "
              f"cov68={cov68[j]:.3f}  cov95={cov95[j]:.3f}")

    row = {"file": fname, "N": len(y_true)}
    for j, name in enumerate(names):
        row[f"mae_{name}"]     = float(mae[j])
        row[f"rmse_{name}"]    = float(rmse[j])
        row[f"bias_{name}"]    = float(bias[j])
        row[f"medsig_{name}"]  = float(med_sigma[j])
        row[f"cov68_{name}"]   = float(cov68[j])
        row[f"cov95_{name}"]   = float(cov95[j])

    # duration bias by bin
    err_dur = err[:, 1]
    print(f"  [bias duration] global: {err_dur.mean():+.3f} h")
    for lo, hi in DUR_BINS:
        m = (y_true[:, 1] >= lo) & (y_true[:, 1] < hi)
        n_bin = int(m.sum())
        if n_bin == 0:
            row[f"dur_bias_{lo}_{hi}"] = np.nan
            row[f"dur_mae_{lo}_{hi}"]  = np.nan
            row[f"dur_n_{lo}_{hi}"]    = 0
            continue
        b = float(err_dur[m].mean())
        ma = float(np.abs(err_dur[m]).mean())
        print(f"    dur [{lo:2d},{hi:2d}) h: bias={b:+.3f}  MAE={ma:.3f}  n={n_bin}")
        row[f"dur_bias_{lo}_{hi}"] = b
        row[f"dur_mae_{lo}_{hi}"]  = ma
        row[f"dur_n_{lo}_{hi}"]    = n_bin

    return row


def _write_csv(rows, out_csv):
    if not rows:
        print("No rows to write.")
        return
    # union of all keys
    keys = ["file", "N"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved summary -> {out_csv}  ({len(rows)} rows, {len(keys)} cols)")


def main():
    print("Using device:", DEVICE)
    model, normalizer, calib = _load_model_and_aux()
    rows = []

    if EVAL_MODE == "data_training":
        ds = TransitNPZDataset(NPZ_PATH, use_time=USE_TIME_CHANNEL, center_flux=CENTER_FLUX)
        n = len(ds)
        _, _, test_idx = make_splits(n, 0.80, 0.10, 0.10, SEED)
        test_loader = DataLoader(
            Subset(ds, test_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        pretty_print_metrics("TEST (uncalibrated)",
                             eval_metrics(model, test_loader, normalizer, DEVICE, calib=None))
        pretty_print_metrics("TEST (calibrated)",
                             eval_metrics(model, test_loader, normalizer, DEVICE, calib=calib))
        y_true, mu, sigma = _predict(test_loader, model, normalizer, calib)
        rows.append(_summarize(NPZ_PATH, y_true, mu, sigma))

    elif EVAL_MODE == "sensitivity":
        npz_files = sorted(glob.glob(os.path.join(SENS_DIR, "sens_*.npz")))
        if not npz_files:
            print(f"No sensitivity .npz files found in '{SENS_DIR}'.")
            return
        print(f"Found {len(npz_files)} sensitivity files in '{SENS_DIR}'.\n")
        for path in npz_files:
            print(f"--- {os.path.basename(path)} ---")
            ds = TransitNPZDataset(path, use_time=USE_TIME_CHANNEL, center_flux=CENTER_FLUX)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            y_true, mu, sigma = _predict(loader, model, normalizer, calib)
            rows.append(_summarize(path, y_true, mu, sigma))

    else:
        raise ValueError(f"Unknown EVAL_MODE '{EVAL_MODE}'.")

    _write_csv(rows, OUT_CSV)


if __name__ == "__main__":
    main()