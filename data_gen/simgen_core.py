import math
import pathlib
import numpy as np
from tqdm import tqdm

try:
    import batman
except ImportError as e:
    raise SystemExit(
        "batman not found. Install with: pip install batman-package\n"
        f"Original error: {e}"
    )


def sample_log_uniform(rng, lo, hi):
    lo = float(lo); hi = float(hi)
    if lo <= 0 or hi <= 0:
        raise ValueError(f"log-uniform requires positive bounds, got lo={lo}, hi={hi}")
    if hi < lo:
        raise ValueError(f"log-uniform requires hi>=lo, got lo={lo}, hi={hi}")
    if hi == lo:
        return lo
    return 10 ** rng.uniform(np.log10(lo), np.log10(hi))


def t14_duration_days(P, a_rs, rp, b):
    #T14, circular orbit

    if b >= (1.0 + rp) or b >= a_rs:
        return np.nan
    sin_i = math.sqrt(max(1e-12, 1.0 - (b / a_rs) ** 2))
    arg = math.sqrt(max(0.0, (1.0 + rp) ** 2 - b ** 2)) / (a_rs * sin_i)
    arg = max(-1.0, min(1.0, arg))
    return (P / math.pi) * math.asin(arg)


def ar1_red_noise(rng, n, sigma, phi=0.90):
    if sigma <= 0:
        return np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    eps = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    s = np.std(x)
    if s > 0:
        x *= (sigma / s)
    return x


def quasi_periodic_stellar(rng, t_rel, sigma_ppm, P_rot_days, ell_days, alpha):
    
    #Quasi-periodic GP draw via Cholesky

    if sigma_ppm <= 0:
        return np.zeros(len(t_rel))
    sigma = sigma_ppm * 1e-6
    dt = t_rel[:, None] - t_rel[None, :]
    K = sigma ** 2 * np.exp(
        -dt ** 2 / (2.0 * ell_days ** 2)
        - np.sin(np.pi * np.abs(dt) / P_rot_days) ** 2 / (2.0 * alpha ** 2)
    )
    K += np.eye(len(t_rel)) * 1e-12
    L_chol = np.linalg.cholesky(K)
    return L_chol @ rng.standard_normal(len(t_rel))


def build_time_grid(cfg):
    cadence_days = cfg["cadence_min"] / (60.0 * 24.0)
    half_w_days = (cfg["window_hours"] / 24.0) / 2.0
    L = int(np.floor((2.0 * half_w_days) / cadence_days)) + 1
    t_rel = (-half_w_days) + cadence_days * np.arange(L, dtype=float)
    return t_rel, L, cadence_days


def make_one_sample(rng, cfg, t_rel, L):
    # depth-controlled radius
    depth_ppm_target = sample_log_uniform(rng, cfg["depth_ppm_min"], cfg["depth_ppm_max"])
    rp = math.sqrt(depth_ppm_target * 1e-6)

    P = rng.uniform(cfg["P_min_days"], cfg["P_max_days"])
    a_rs = sample_log_uniform(rng, cfg["a_rs_min"], cfg["a_rs_max"])

    # impact parameter with duration constraints
    dur = np.nan
    b = np.nan
    for _ in range(800):
        b_try = rng.uniform(0.0, cfg["b_max"])
        if b_try >= (1.0 + rp) or b_try >= a_rs:
            continue
        dur_try = t14_duration_days(P, a_rs, rp, b_try)
        if not np.isfinite(dur_try):
            continue
        if dur_try < cfg["dur_min_days"] or dur_try > cfg["dur_max_days"]:
            continue
        b, dur = b_try, dur_try
        break
    if not np.isfinite(dur):
        return None

    inc = math.degrees(math.acos(max(-1.0, min(1.0, b / a_rs))))

    #u1 = rng.uniform(cfg["u1_min"], cfg["u1_max"])
    #u2 = rng.uniform(cfg["u2_min"], cfg["u2_max"])

    while True:
        q1, q2 = rng.uniform(), rng.uniform()
        sq = math.sqrt(q1)
        u1 = 2 * sq * q2
        u2 = sq * (1 - 2 * q2)
        if cfg["u1_min"] <= u1 <= cfg["u1_max"] and cfg["u2_min"] <= u2 <= cfg["u2_max"]:
            break

    # epoch offset
    off_h = rng.uniform(-cfg["max_epoch_offset_hours"], +cfg["max_epoch_offset_hours"])
    offset_days = off_h / 24.0

    t0_true = 0.0
    t0_hat = t0_true + offset_days
    t_abs = t0_hat + t_rel

    # batman
    params = batman.TransitParams()
    params.t0 = t0_true
    params.per = P
    params.rp = rp
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = [u1, u2]

    m = batman.TransitModel(
        params, t_abs,
        supersample_factor=cfg["supersample_factor"],
        exp_time=cfg["exp_time_days"],
    )
    flux_model = m.light_curve(params)

    depth = 1.0 - float(np.min(flux_model))
    duration_days = float(dur)
    dt_correction_days = float(t0_true - t0_hat)

    # baseline
    slope_ppm = rng.uniform(-cfg["slope_ppm_per_day"], cfg["slope_ppm_per_day"])
    quad_ppm = rng.uniform(-cfg["quad_ppm_per_day2"], cfg["quad_ppm_per_day2"])
    baseline = 1.0 + (slope_ppm * t_rel + quad_ppm * (t_rel ** 2)) * 1e-6

    # white noise
    sigma_w_ppm = sample_log_uniform(rng, cfg["sigma_w_ppm_min"], cfg["sigma_w_ppm_max"])

    # reject samples (snr)
    min_snr = cfg.get("min_snr_integrated", 3.0)
    cadence_days = cfg["cadence_min"] / (60.0 * 24.0)
    n_in_transit = max(duration_days / cadence_days, 1.0)
    snr = depth_ppm_target / sigma_w_ppm * math.sqrt(n_in_transit)
    if snr < min_snr:
        return None

    white = rng.normal(0.0, sigma_w_ppm * 1e-6, size=L)

    # red noise
    p_no_red = cfg.get("p_no_red", 0.50)
    if rng.random() < p_no_red or cfg["sigma_r_ppm_max"] <= 0:
        sigma_r_ppm = 0.0
    else:
        sigma_r_ppm = sample_log_uniform(rng, cfg["sigma_r_ppm_min_pos"], cfg["sigma_r_ppm_max"])
    red = ar1_red_noise(rng, L, sigma_r_ppm * 1e-6, phi=cfg["ar1_phi"])

    # stellar variability (quasi-periodic GP)
    p_no_stellar = cfg.get("p_no_stellar", 1.0)
    if rng.random() >= p_no_stellar and cfg.get("sigma_stellar_ppm_max", 0.0) > 0:
        sigma_stellar_ppm = sample_log_uniform(
            rng, cfg["sigma_stellar_ppm_min"], cfg["sigma_stellar_ppm_max"]
        )
        P_rot = rng.uniform(cfg["P_rot_min_days"], cfg["P_rot_max_days"])
        ell_days = rng.uniform(cfg["ell_days_min"], cfg["ell_days_max"])
        alpha_stellar = rng.uniform(cfg["alpha_stellar_min"], cfg["alpha_stellar_max"])
        stellar = quasi_periodic_stellar(
            rng, t_rel, sigma_stellar_ppm, P_rot, ell_days, alpha_stellar,
        )
    else:
        sigma_stellar_ppm = 0.0
        P_rot = 0.0
        ell_days = 0.0
        alpha_stellar = 0.0
        stellar = np.zeros(L)

    flux = flux_model * baseline + white + red + stellar

    # outliers
    if cfg["outlier_prob"] > 0:
        mask = rng.random(L) < cfg["outlier_prob"]
        if np.any(mask):
            flux[mask] += rng.normal(
                0.0, cfg["outlier_sigma_ppm"] * 1e-6, size=int(np.sum(mask))
            )

    # normalize
    med = np.median(flux)
    if med == 0 or not np.isfinite(med):
        return None
    flux = flux / med

    X = np.stack([t_rel.astype(np.float32), flux.astype(np.float32)], axis=-1)
    y = np.array([depth, duration_days, dt_correction_days], dtype=np.float32)

    meta = dict(
        depth_ppm_target=depth_ppm_target,
        rp=rp, a_rs=a_rs, b=b, inc_deg=inc, u1=u1, u2=u2,
        sigma_w_ppm=sigma_w_ppm, sigma_r_ppm=sigma_r_ppm,
        sigma_stellar_ppm=sigma_stellar_ppm, P_rot_days=P_rot,
        ell_days=ell_days,               
        alpha_stellar=alpha_stellar,    
        slope_ppm_per_day=slope_ppm, quad_ppm_per_day2=quad_ppm,
        offset_days=offset_days,
        P_days=P,
        snr_white=snr, 
    )
    return X, y, meta


# Dataset generation
def generate_dataset(n, seed, cfg, desc="Accepted samples"):
    rng = np.random.default_rng(seed)
    t_rel, L, _ = build_time_grid(cfg)

    X_list, y_list = [], []
    meta_cols = {
        "depth_ppm_target": [],
        "rp": [], "a_rs": [], "b": [], "inc_deg": [], "u1": [], "u2": [],
        "sigma_w_ppm": [], "sigma_r_ppm": [],
        "sigma_stellar_ppm": [], "P_rot_days": [],
        "slope_ppm_per_day": [], "quad_ppm_per_day2": [],
        "offset_days": [], "P_days": [],
        "ell_days": [], "alpha_stellar": [], "snr_white": [],
    }

    attempts = 0
    pbar = tqdm(total=n, desc=desc, unit="sample")

    while len(X_list) < n:
        attempts += 1
        out = make_one_sample(rng, cfg, t_rel, L)
        if out is None:
            if attempts % 2000 == 0:
                rej = 1.0 - (len(X_list) / attempts)
                pbar.set_postfix({"attempts": attempts, "reject%": f"{100*rej:.1f}%"})
            continue

        X, y, meta = out
        X_list.append(X)
        y_list.append(y)
        for k in meta_cols:
            meta_cols[k].append(meta[k])

        pbar.update(1)

        if len(X_list) % 500 == 0:
            rej = 1.0 - (len(X_list) / attempts)
            pbar.set_postfix({"attempts": attempts, "reject%": f"{100*rej:.1f}%"})

        if attempts > n * 60:
            pbar.close()
            raise RuntimeError("Too many rejected samples. Relax cfg bounds.")

    pbar.close()
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    meta = {k: np.array(v) for k, v in meta_cols.items()}
    return X, y, meta


def save_npz(path, X, y, meta, cfg):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X, y=y,
        **{f"meta_{k}": v for k, v in meta.items()},
        cfg=np.array([cfg], dtype=object),
    )