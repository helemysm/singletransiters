import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import data_gen.make_sensitivity_sets as make_sensitivity_sets
from core.config import cfg
from core.config import cfg, resolve_log_mask
import os
import re
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from core.data import TransitNPZDataset
from core.normalizer import TargetNormalizer
from core.calibrate import SigmaCalibrator


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


RUN_SENSITIVITY_TEST = True

SENS_DIR    = cfg["paths"]["sensitivity_sets"]
MODEL_PATH  = cfg["paths"]["model"]
AUX_PATH    = cfg["paths"]["aux"]
OUTPUT_DIR  = "sentitivity_plots_cnn_v9"

BATCH_SIZE       = 512
USE_TIME_CHANNEL = True
CENTER_FLUX      = True
FAIL_THRESH_H = 5.0  

PARAM_NAMES   = ["depth_ppm", "duration_h", "dtcorr_h"]
PARAM_LABELS  = ["Depth", "Duration", r"$\Delta t$"]
PARAM_UNITS   = ["ppm", "hours", "hours"]
PARAM_COLORS  = ["#0072B2", "#E69F00", "#785EF0"]


if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def load_model_and_aux():
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


# Evaluation on one .npz
def evaluate_npz(npz_path, model, normalizer, calib):
    ds = TransitNPZDataset(npz_path, use_time=USE_TIME_CHANNEL, center_flux=CENTER_FLUX)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_true_list, mu_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mu_n, log_sigma_n = model(x)
            log_sigma_n = calib(log_sigma_n)
            mu, _ = normalizer.decode_mu_sigma(mu_n, log_sigma_n)
            y_true_list.append(y.cpu().numpy())
            mu_list.append(mu.cpu().numpy())

    y_true = np.vstack(y_true_list)
    mu     = np.vstack(mu_list)
    err    = mu - y_true
    mae    = np.mean(np.abs(err), axis=0)
    rmse   = np.sqrt(np.mean(err ** 2, axis=0))
    fail_rate_dt = np.mean(np.abs(err[:, 2]) > FAIL_THRESH_H) * 100.0

    # SNR summary
    
    d = np.load(npz_path, allow_pickle=True)
    if "meta_sigma_w_ppm" in d.files:
        sigma_w_med = float(np.median(d["meta_sigma_w_ppm"]))
    else:
        sigma_w_med = np.nan

    if "meta_snr_white" in d.files:
        snr = d["meta_snr_white"].astype(float)
        snr_stats = (np.median(snr), np.percentile(snr, 10), np.percentile(snr, 90))
    else:
        snr_stats = (np.nan, np.nan, np.nan)

    return mae, rmse, snr_stats, fail_rate_dt, sigma_w_med


# parser
SWEEP_PATTERNS = {
    "noise":   (re.compile(r"sens_noise_sigma(\d+)ppm"),   r"White-noise $\sigma_w$ [ppm]"),
    "offset":  (re.compile(r"sens_offset_off(\d+)h"),      "Epoch offset [h]"),
    "morph":   (re.compile(r"sens_morph_bmax(\d+\.\d+)"),  r"$b_\mathrm{max}$"),
    "stellar": (re.compile(r"sens_stellar_sigma(\d+)ppm"), r"Stellar variability $\sigma_\star$ [ppm]"),
    "red":     (re.compile(r"sens_red_sigma(\d+)ppm"),     r"Red noise $\sigma_r$ [ppm]"),
}


def parse_files(npz_files):
    sweeps = {k: [] for k in SWEEP_PATTERNS}
    for path in npz_files:
        fname = os.path.basename(path)
        for key, (pat, _) in SWEEP_PATTERNS.items():
            m = pat.search(fname)
            if m:
                level = float(m.group(1))
                sweeps[key].append((level, path))
                break
    for key in sweeps:
        sweeps[key].sort(key=lambda t: t[0])
    return sweeps


def plot_sweep_per_param(sweep_name, xlabel, levels, mae_matrix, rmse_matrix):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f"Sensitivity: {sweep_name} sweep", fontsize=14)

    for j, (label, unit, color) in enumerate(zip(PARAM_LABELS, PARAM_UNITS, PARAM_COLORS)):
        axes[j].plot(levels, mae_matrix[:, j], marker="o", color=color, lw=2)
        axes[j].set_title(f"MAE — {label}")
        axes[j].set_ylabel(f"MAE [{unit}]")
        axes[j].set_xlabel(xlabel)
        axes[j].grid(True, alpha=0.3)
        axes[j].tick_params(axis='both', direction='in')

    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"sensitivity_{sweep_name}_metrics.png")
    fig.savefig(save_path, dpi=200)
    print(f"  -> Saved metric plot to: {save_path}")
    plt.close(fig)


def plot_sweep_normalized(sweep_name, xlabel, levels, mae_matrix):
    
    ref = mae_matrix[0, :]
    ref_safe = np.where(ref > 0, ref, 1.0)
    rel = mae_matrix / ref_safe[None, :]

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, (label, color) in enumerate(zip(PARAM_LABELS, PARAM_COLORS)):
        ax.plot(levels, rel[:, j], marker="o", lw=2, color=color, label=label)

    ax.axhline(1.0, ls="--", color="gray", alpha=0.6, label="reference (level 0)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative MAE (ratio to reference level)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', direction='in')

    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"sensitivity_{sweep_name}_normalized.png")
    fig.savefig(save_path, dpi=200)
    print(f"  -> Saved summary plot to: {save_path}")
    plt.close(fig)



def main():
    if not RUN_SENSITIVITY_TEST:
        print("RUN_SENSITIVITY_TEST = False — skipping sensitivity analysis.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    expected = (
        [f"sens_noise_sigma{int(s)}ppm.npz"    for s in make_sensitivity_sets.NOISE_LEVELS_PPM] +
        [f"sens_offset_off{int(o)}h.npz"       for o in make_sensitivity_sets.OFFSET_LEVELS_H] +
        [f"sens_morph_bmax{b:.2f}.npz"         for b in make_sensitivity_sets.BMAX_LEVELS] +
        [f"sens_stellar_sigma{int(s)}ppm.npz"  for s in make_sensitivity_sets.STELLAR_LEVELS_PPM] +
        [f"sens_red_sigma{int(s)}ppm.npz"      for s in make_sensitivity_sets.RED_LEVELS_PPM]
    )
    missing = [f for f in expected if not os.path.exists(os.path.join(SENS_DIR, f))]
    if missing:
        print("=" * 60)
        print(f"Step 1: Generating {len(missing)} missing sensitivity datasets …")
        print("=" * 60)
        make_sensitivity_sets.main()
    else:
        print("Step 1: All sensitivity files already exist, skipping generation.")

    npz_files = sorted(glob.glob(os.path.join(SENS_DIR, "sens_*.npz")))
    if not npz_files:
        print(f"No sensitivity .npz files in '{SENS_DIR}'. Aborting.")
        return
    print(f"\nFound {len(npz_files)} sensitivity files.")

    print("\nStep 2: Loading model …")
    model, normalizer, calib = load_model_and_aux()
    print(f"  Device: {DEVICE}")

    print("\nStep 3: Evaluating …")
    results = {}
    for path in npz_files:
        fname = os.path.basename(path)
        mae, rmse, snr_stats, fail_dt, sigma_w_med = evaluate_npz(path, model, normalizer, calib)
        results[path] = (mae, rmse, snr_stats, fail_dt, sigma_w_med)
        snr_med, snr_p10, snr_p90 = snr_stats
        print(
            f"  {fname}: "
            f"σ_w={sigma_w_med:6.0f} ppm | "
            f"SNR={snr_med:5.1f} [{snr_p10:4.1f}-{snr_p90:5.1f}] | "
            f"MAE_depth={mae[0]:7.2f} ppm | "
            f"MAE_dur={mae[1]:.3f} h | "
            f"MAE_dt={mae[2]:.3f} h | "
            f"fail_dt={fail_dt:5.1f}%"
        )

    print("\nStep 4: Plotting and saving results …")
    sweeps = parse_files(npz_files)

    for sweep_name, (_, xlabel) in SWEEP_PATTERNS.items():
        entries = sweeps[sweep_name]
        if not entries:
            print(f"  No files for sweep '{sweep_name}', skipping.")
            continue

        levels      = np.array([lvl for lvl, _ in entries])
        mae_matrix  = np.array([results[p][0] for _, p in entries])
        rmse_matrix = np.array([results[p][1] for _, p in entries])

        plot_sweep_per_param(sweep_name, xlabel, levels, mae_matrix, rmse_matrix)
        plot_sweep_normalized(sweep_name, xlabel, levels, mae_matrix)


if __name__ == "__main__":
    main()