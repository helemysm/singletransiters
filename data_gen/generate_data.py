import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from data_gen.simgen_core import generate_dataset, save_npz
from core.config import cfg

OUT_PATH  = "data/test_realistic_2.npz"#cfg["paths"]["train_data"]
N_SAMPLES = 20000#cfg["data_generation"]["n_samples"]
SEED      = 999#cfg["data_generation"]["seed"]


def main():
    cfg = dict(
        
        window_hours=99.5,
        cadence_min=30.0,
        supersample_factor=15,
        exp_time_days=30.0 / (60.0 * 24.0),

        max_epoch_offset_hours=35.0,

        depth_ppm_min=300,#1200.0,
        depth_ppm_max=15000.0,#15000.0,

        P_min_days=12.0,
        P_max_days=200.0,

        a_rs_min=8.0,
        a_rs_max=60.0,

        b_max=0.90,

        dur_min_days=1.5 / 24.0,
        dur_max_days=15.0 / 24.0,

        u1_min=0.1, u1_max=0.6,
        u2_min=0.0, u2_max=0.5,

        sigma_w_ppm_min=50.0,
        sigma_w_ppm_max=2000.0, #1500
        min_snr_integrated=3,  # reject depth/noise 

        #red noise ----
        p_no_red=0.45,#0.60,
        sigma_r_ppm_min_pos=10.0,
        sigma_r_ppm_max=200.0,
        ar1_phi=0.95,#0.85,

        # baseline polynomial trend 
        slope_ppm_per_day=200.0,
        quad_ppm_per_day2=20.0,

        # outliers 
        outlier_prob=0.0005,
        outlier_sigma_ppm=5000.0,

        #  stellar variability (quasi-periodic GP) 
        
        p_no_stellar=0.15,
        sigma_stellar_ppm_min=50.0,
        sigma_stellar_ppm_max=2000,#1500.0, #
        P_rot_min_days=2.0,     # fast rotators matter for TESS
        P_rot_max_days=30.0,
        ell_days=2.0,           
        alpha_stellar=0.5,      # periodicity strength
        ell_days_min=0.5,                
        ell_days_max=10,#5.0,               
        alpha_stellar_min=0.3,        
        alpha_stellar_max=1.0, 
    )

    X, y, meta = generate_dataset(N_SAMPLES, SEED, cfg, desc="training")
    save_npz(OUT_PATH, X, y, meta, cfg)
    print("Saved:", OUT_PATH, "X", X.shape, "y", y.shape)


if __name__ == "__main__":
    main()