import os
from data_gen.simgen_core import generate_dataset, save_npz
import core.config as _config


OUT_DIR = _config.cfg["paths"]["sensitivity_sets"]
os.makedirs(OUT_DIR, exist_ok=True)

SEED_BASE = 123
N_PER_SET = 2000  

CADENCE_MIN = 30.0
WINDOW_HOURS = 99.5
SUPERSAMPLE = 15

DEPTH_PPM_MIN = 300.0
DEPTH_PPM_MAX = 15000 #8000.0

DUR_MIN_H = 1.5
DUR_MAX_H = 15.0

P_MIN = 12.0
P_MAX = 200.0
A_RS_MIN = 8.0
A_RS_MAX = 60.0

SLOPE_PPM_PER_DAY = 200.0
QUAD_PPM_PER_DAY2 = 20.0

DEFAULT_SIGMA_W_MIN = 50#100.0
DEFAULT_SIGMA_W_MAX = 3000#800.0

DEFAULT_P_NO_RED = 0.60
DEFAULT_SIGMA_R_MIN = 10.0
DEFAULT_SIGMA_R_MAX = 200.0
AR1_PHI = 0.85

DEFAULT_P_NO_STELLAR = 0.30
DEFAULT_SIGMA_STELLAR_MIN = 50.0
DEFAULT_SIGMA_STELLAR_MAX = 1500#700.0
DEFAULT_P_ROT_MIN = 2.0
DEFAULT_P_ROT_MAX = 30.0
DEFAULT_ELL_DAYS = 2.0
DEFAULT_ELL_DAYS_MIN = 0.5
DEFAULT_ELL_DAYS_MAX = 5.0
DEFAULT_ALPHA_STELLAR = 0.5
DEFAULT_ALPHA_STELLAR_MIN = 0.3
DEFAULT_ALPHA_STELLAR_MAX = 1.0

DEFAULT_B_MAX = 0.90

OUTLIER_PROB = 0.0005
OUTLIER_SIGMA_PPM = 5000.0


#NOISE_LEVELS_PPM   = [50, 100, 200, 400, 800, 1200, 2000, 3000]
NOISE_LEVELS_PPM   = [50, 100, 200, 400, 800, 1200, 1500, 2000, 3000]
OFFSET_LEVELS_H    = [0, 2, 6, 12, 24, 36, 40]
BMAX_LEVELS        = [0.10, 0.30, 0.50, 0.70, 0.85, 0.95]
STELLAR_LEVELS_PPM = [0, 100, 300, 500, 700, 1000, 1500]
RED_LEVELS_PPM     = [0, 20, 50, 100, 150, 200]


def base_cfg():
    
    
    return dict(
        window_hours=float(WINDOW_HOURS),
        cadence_min=float(CADENCE_MIN),
        supersample_factor=int(SUPERSAMPLE),
        exp_time_days=float(CADENCE_MIN) / (60.0 * 24.0),

        max_epoch_offset_hours=35.0,

        depth_ppm_min=float(DEPTH_PPM_MIN),
        depth_ppm_max=float(DEPTH_PPM_MAX),

        P_min_days=float(P_MIN),
        P_max_days=float(P_MAX),
        a_rs_min=float(A_RS_MIN),
        a_rs_max=float(A_RS_MAX),

        b_max=float(DEFAULT_B_MAX),

        dur_min_days=float(DUR_MIN_H) / 24.0,
        dur_max_days=float(DUR_MAX_H) / 24.0,

        u1_min=0.1, u1_max=0.6,
        u2_min=0.0, u2_max=0.5,

        min_snr_integrated=2, # lower but not zero

        sigma_w_ppm_min=float(DEFAULT_SIGMA_W_MIN),
        sigma_w_ppm_max=float(DEFAULT_SIGMA_W_MAX),

        p_no_red=float(DEFAULT_P_NO_RED),
        sigma_r_ppm_min_pos=float(DEFAULT_SIGMA_R_MIN),
        sigma_r_ppm_max=float(DEFAULT_SIGMA_R_MAX),
        ar1_phi=float(AR1_PHI),

        # trends
        slope_ppm_per_day=float(SLOPE_PPM_PER_DAY),
        quad_ppm_per_day2=float(QUAD_PPM_PER_DAY2),
        # outliers
        outlier_prob=float(OUTLIER_PROB),
        outlier_sigma_ppm=float(OUTLIER_SIGMA_PPM),

        p_no_stellar=float(DEFAULT_P_NO_STELLAR),
        sigma_stellar_ppm_min=float(DEFAULT_SIGMA_STELLAR_MIN),
        sigma_stellar_ppm_max=float(DEFAULT_SIGMA_STELLAR_MAX),
        P_rot_min_days=float(DEFAULT_P_ROT_MIN),
        P_rot_max_days=float(DEFAULT_P_ROT_MAX),
        ell_days=float(DEFAULT_ELL_DAYS),
        ell_days_min=float(DEFAULT_ELL_DAYS_MIN),
        ell_days_max=float(DEFAULT_ELL_DAYS_MAX),
        alpha_stellar=float(DEFAULT_ALPHA_STELLAR),
        alpha_stellar_min=float(DEFAULT_ALPHA_STELLAR_MIN),
        alpha_stellar_max=float(DEFAULT_ALPHA_STELLAR_MAX),
    )


def run_noise_sweep():
    
    for k, sig in enumerate(NOISE_LEVELS_PPM):
        cfg = base_cfg()
        cfg["sigma_w_ppm_min"] = float(sig)
        cfg["sigma_w_ppm_max"] = float(sig)
        # isolate white noise
        cfg["p_no_red"] = 1.0
        cfg["sigma_r_ppm_max"] = 0.0
        cfg["p_no_stellar"] = 1.0
        cfg["sigma_stellar_ppm_max"] = 0.0

        out_path = os.path.join(OUT_DIR, f"sens_noise_sigma{int(sig)}ppm.npz")
        X, y, meta = generate_dataset(
            N_PER_SET, SEED_BASE + 10 + k, cfg, desc=f"noise={sig}ppm"
        )
        save_npz(out_path, X, y, meta, cfg)
        print("Saved:", out_path)


def run_offset_sweep():
    
    print("=== Epoch-offset sweep (training-like noise) ===")
    for k, off in enumerate(OFFSET_LEVELS_H):
        cfg = base_cfg()
        cfg["max_epoch_offset_hours"] = float(off)

        out_path = os.path.join(OUT_DIR, f"sens_offset_off{int(off)}h.npz")
        X, y, meta = generate_dataset(
            N_PER_SET, SEED_BASE + 100 + k, cfg, desc=f"offset<=±{off}h"
        )
        save_npz(out_path, X, y, meta, cfg)
        print("Saved:", out_path)


def run_morphology_sweep():
    
    print("=== Morphology (b_max) sweep (training-like noise) ===")
    for k, bmax in enumerate(BMAX_LEVELS):
        cfg = base_cfg()
        cfg["b_max"] = float(bmax)

        out_path = os.path.join(OUT_DIR, f"sens_morph_bmax{bmax:.2f}.npz")
        X, y, meta = generate_dataset(
            N_PER_SET, SEED_BASE + 200 + k, cfg, desc=f"b_max={bmax:.2f}"
        )
        save_npz(out_path, X, y, meta, cfg)
        print("Saved:", out_path)


def run_stellar_sweep():
    
    print("=== Stellar variability sweep ===")
    for k, sig_s in enumerate(STELLAR_LEVELS_PPM):
        cfg = base_cfg()
        cfg["p_no_red"] = 1.0
        cfg["sigma_r_ppm_max"] = 0.0

        if sig_s <= 0:
            cfg["p_no_stellar"] = 1.0
            cfg["sigma_stellar_ppm_max"] = 0.0
        else:
            cfg["p_no_stellar"] = 0.0  # 100% of samples have variability
            cfg["sigma_stellar_ppm_min"] = float(sig_s)
            cfg["sigma_stellar_ppm_max"] = float(sig_s) 

        out_path = os.path.join(OUT_DIR, f"sens_stellar_sigma{int(sig_s)}ppm.npz")
        X, y, meta = generate_dataset(
            N_PER_SET, SEED_BASE + 300 + k, cfg, desc=f"stellar={sig_s}ppm"
        )
        save_npz(out_path, X, y, meta, cfg)
        print("Saved:", out_path)


def run_red_sweep():
    
    print("=== Red noise sweep ===")
    for k, sig_r in enumerate(RED_LEVELS_PPM):
        cfg = base_cfg()
        
        cfg["p_no_stellar"] = 1.0
        cfg["sigma_stellar_ppm_max"] = 0.0

        if sig_r <= 0:
            cfg["p_no_red"] = 1.0
            cfg["sigma_r_ppm_max"] = 0.0
        else:
            cfg["p_no_red"] = 0.0  
            cfg["sigma_r_ppm_min_pos"] = float(sig_r)
            cfg["sigma_r_ppm_max"] = float(sig_r)  # fixed level

        out_path = os.path.join(OUT_DIR, f"sens_red_sigma{int(sig_r)}ppm.npz")
        X, y, meta = generate_dataset(
            N_PER_SET, SEED_BASE + 400 + k, cfg, desc=f"red={sig_r}ppm"
        )
        save_npz(out_path, X, y, meta, cfg)
        print("Saved:", out_path)


def main():
    total = (len(NOISE_LEVELS_PPM) + len(OFFSET_LEVELS_H) + len(BMAX_LEVELS)
             + len(STELLAR_LEVELS_PPM) + len(RED_LEVELS_PPM))
    print(f"Generating {total} sensitivity sets × {N_PER_SET} samples each")
    print(f"Total synthetic light curves: {total * N_PER_SET:,}")
    run_noise_sweep()
    run_offset_sweep()
    run_morphology_sweep()
    run_stellar_sweep()
    run_red_sweep()


if __name__ == "__main__":
    main()