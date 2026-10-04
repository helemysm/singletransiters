import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import lightkurve as lk

from core.config import cfg
from core.normalizer import TargetNormalizer
from core.calibrate import SigmaCalibrator

warnings.filterwarnings("ignore", category=lk.utils.LightkurveDeprecationWarning)

MODEL_PATH = cfg["paths"]["model"]
AUX_PATH   = cfg["paths"]["aux"]

WINDOW_HOURS = 99.5
CADENCE_MIN  = 30.0

DO_BIN   = True
BIN_DAYS = CADENCE_MIN / (60.0 * 24.0)

USE_TIME_CHANNEL = True
CENTER_FLUX      = True


def pick_lightcurve(obj):
    
    for attr in ("PDCSAP_FLUX", "SAP_FLUX"):
        try:
            return getattr(obj, attr)
        except (KeyError, AttributeError):
            continue
    return obj


def preprocess_lc(lc):
    
    if hasattr(lc, "quality") and lc.quality is not None:
        lc = lc[lc.quality == 0]
    lc = lc.remove_nans()
    if DO_BIN:
        lc = lc.bin(time_bin_size=BIN_DAYS).remove_nans()
    return lc


def build_fixed_window(t, f, t0_hat, window_hours=99.5, cadence_min=30.0):
    order = np.argsort(t)
    t = t[order]
    f = f[order]

    cadence_days = cadence_min / (60.0 * 24.0)
    half_w_days  = (window_hours / 24.0) / 2.0
    L = int(np.floor((2.0 * half_w_days) / cadence_days)) + 1

    t_rel  = (-half_w_days) + cadence_days * np.arange(L, dtype=float)
    t_grid = t0_hat + t_rel

    med    = np.nanmedian(f)
    f_grid = np.interp(t_grid, t, f, left=med, right=med)

    med2 = np.nanmedian(f_grid)
    if (not np.isfinite(med2)) or (med2 == 0.0):
        raise RuntimeError("Bad median for normalization in the selected window.")
    f_grid = f_grid / med2

    return t_rel, f_grid


def to_model_input(t_rel_days, flux, center_flux=True, use_time=True):
    t_h        = t_rel_days * 24.0
    half_range = max(0.5 * (np.max(t_h) - np.min(t_h)), 1e-6)
    t_scaled   = (t_h / half_range).astype(np.float32)

    flux = flux.astype(np.float32)
    if center_flux:
        flux = flux - 1.0

    x = np.stack([t_scaled, flux], axis=0) if use_time else flux[None, :]
    return torch.from_numpy(x)[None, :, :]


def load_bundle(model_path, aux_path, device, expected_use_time=True):
    aux        = np.load(aux_path, allow_pickle=True)
    log_mask   = aux["y_log_mask"] if "y_log_mask" in aux.files else None
    normalizer = TargetNormalizer(mean=aux["y_mean"], std=aux["y_std"], log_mask=log_mask)

    calib = SigmaCalibrator(d_out=3)
    calib.log_scale.data = torch.tensor(aux["calib_log_scale"], dtype=torch.float32)
    calib = calib.to(device).eval()

    ckpt     = torch.load(model_path, map_location="cpu")
    in_ch    = 2 if ckpt.get("use_time_channel", expected_use_time) else 1
    model    = cfg["model_class"](
        in_ch=in_ch, d_out=3,
        log_sigma_min=ckpt.get("log_sigma_min", -8.0),
        log_sigma_max=ckpt.get("log_sigma_max",  4.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    return model, calib, normalizer, ckpt


@torch.no_grad()
def predict_one(model, calib, normalizer, x, device):
    x = x.to(device)
    mu_n, log_sigma_n = model(x)
    log_sigma_n = calib(log_sigma_n)
    mu, sigma   = normalizer.decode_mu_sigma(mu_n, log_sigma_n)
    return mu.squeeze(0).cpu().numpy(), sigma.squeeze(0).cpu().numpy()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_t0_interactively(t, f_norm):
    
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.plot(t, f_norm, lw=0.6, color="steelblue")
    ax.set_xlabel("BTJD (BJD − 2457000)")
    ax.set_ylabel("normalised flux")
    ax.set_title("Click on the transit center — close window to confirm")
    ax.grid(True, alpha=0.2)

    clicked = []
    vline   = [None]

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        clicked.clear()
        clicked.append(event.xdata)
        if vline[0] is not None:
            vline[0].remove()
        vline[0] = ax.axvline(event.xdata, color="red", lw=1.5, ls="--",
                              label=f"t₀ = {event.xdata:.4f}")
        ax.legend(fontsize=9)
        ax.set_title(f"Selected t0_hat = {event.xdata:.4f} BTJD — close to confirm")
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    return clicked[-1] if clicked else None

#####core
def _run_inference_on_lc(lc, t0_hat, plot=True):
    """Run inference on an already-loaded, preprocessed LC. Returns a result dict."""
    device = get_device()

    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)

    print("TIME range:", float(np.nanmin(t)), "to", float(np.nanmax(t)))
    if not (np.nanmin(t) <= t0_hat <= np.nanmax(t)):
        print("WARNING: t0_hat is outside the TIME range; window will be filled with median.")

    t_rel, f_grid = build_fixed_window(t, f, t0_hat, window_hours=WINDOW_HOURS, cadence_min=CADENCE_MIN)
    x = to_model_input(t_rel, f_grid, center_flux=CENTER_FLUX, use_time=USE_TIME_CHANNEL)

    model, calib, normalizer, ckpt = load_bundle(MODEL_PATH, AUX_PATH, device, expected_use_time=USE_TIME_CHANNEL)
    if ckpt.get("use_time_channel", USE_TIME_CHANNEL) != USE_TIME_CHANNEL:
        print(f"WARNING: ckpt use_time_channel != USE_TIME_CHANNEL={USE_TIME_CHANNEL}")

    mu, sig = predict_one(model, calib, normalizer, x, device)
    depth_ppm, duration_h, dtcorr_h = mu
    s_depth,   s_dur,      s_dt     = sig

    t_mid_pred_btjd   = t0_hat + (dtcorr_h / 24.0)
    t_start_pred_btjd = t_mid_pred_btjd - (duration_h / 2.0) / 24.0
    t_end_pred_btjd   = t_mid_pred_btjd + (duration_h / 2.0) / 24.0
    t_mid_pred_bjd    = t_mid_pred_btjd   + 2457000.0
    t_start_pred_bjd  = t_start_pred_btjd + 2457000.0
    t_end_pred_bjd    = t_end_pred_btjd   + 2457000.0

    print(f"\nPrediction (calibrated):")
    print(f"  depth_ppm : {depth_ppm:.1f} ± {s_depth:.1f}")
    print(f"  duration_h: {duration_h:.3f} ± {s_dur:.3f}")
    print(f"  dtcorr_h  : {dtcorr_h:.3f} ± {s_dt:.3f}")
    print(f"  t_mid_pred   (BTJD): {t_mid_pred_btjd:.8f}")
    print(f"  t_start_pred (BTJD): {t_start_pred_btjd:.8f}")
    print(f"  t_end_pred   (BTJD): {t_end_pred_btjd:.8f}")
    print(f"  t_mid_pred   (BJD) : {t_mid_pred_bjd:.8f}")

    if plot:
        t_abs_btjd = t0_hat + t_rel
        plt.figure(figsize=(11, 4))
        plt.plot(t_abs_btjd, f_grid, lw=0.8, alpha=0.6)
        plt.scatter(t_abs_btjd, f_grid, s=6, alpha=0.7, linewidths=0)
        plt.axvline(t_start_pred_btjd, ls=":", lw=1, label="pred t_start (BTJD)")
        plt.axvline(t_end_pred_btjd,   ls=":", lw=1, label="pred t_end (BTJD)")
        plt.axvline(t0_hat,            ls="--", lw=1, label=f"t0_hat = {t0_hat:.6f} BTJD")
        plt.axvline(t_mid_pred_btjd,   ls="-.", lw=2,
                    label=f"pred t0_true = {t_mid_pred_btjd:.6f} BTJD")
        plt.title(
            f"depth={depth_ppm:.0f}±{s_depth:.0f} ppm | "
            f"dur={duration_h:.2f}±{s_dur:.2f} h | "
            f"t0_pred={t_mid_pred_btjd:.6f}±{s_dt / 24.0:.6f} BTJD"
        )
        plt.xlabel("BTJD (BJD − 2457000)")
        plt.ylabel("flux normalized")
        plt.tight_layout()
        plt.legend()
        plt.show()

    return {
        "depth_ppm":          float(depth_ppm),
        "duration_h":         float(duration_h),
        "dtcorr_h":           float(dtcorr_h),
        "sigma_depth":        float(s_depth),
        "sigma_duration":     float(s_dur),
        "sigma_dtcorr":       float(s_dt),
        "t_mid_pred_btjd":    float(t_mid_pred_btjd),
        "t_start_pred_btjd":  float(t_start_pred_btjd),
        "t_end_pred_btjd":    float(t_end_pred_btjd),
        "t_mid_pred_bjd":     float(t_mid_pred_bjd),
        "t_start_pred_bjd":   float(t_start_pred_bjd),
        "t_end_pred_bjd":     float(t_end_pred_bjd),
        "t_rel_days":         t_rel,
        "flux_window":        f_grid,
    }


#Load a light curve file/URL and run inferenc   
def run_inference_for_file(path_to_file, t0_hat, plot=True):
    obj = lk.read(path_to_file)
    lc  = pick_lightcurve(obj)
    lc  = preprocess_lc(lc)
    return _run_inference_on_lc(lc, t0_hat, plot=plot)
