import argparse
from pathlib import Path

import pandas as pd
import lightkurve as lk

from vetting.param_inference_core import pick_lightcurve, preprocess_lc, _run_inference_on_lc
from vetting.compute_param_single import _planet_radius_rjup, _rjup_to_rearth

REQUIRED_COLUMNS = ["tic", "sector", "t0"]
RESULT_COLUMNS = [
    "depth_ppm", "sigma_depth",
    "duration_h", "sigma_duration",
    "dtcorr_h", "sigma_dtcorr",
    "t_mid_pred_btjd", "radius_rjup", "radius_rearth",
    "error",
]


def _download_sector(tic, sector):
    sr = lk.search_lightcurve(f"TIC {tic}", sector=sector, author="SPOC")
    if len(sr) == 0:
        raise RuntimeError(f"no SPOC light curve for sector {sector}")
    lc = pick_lightcurve(sr[0].download())
    return preprocess_lc(lc)


def main():
    parser = argparse.ArgumentParser(description="Single-transit parameter inference from a CSV (tic, sector, t0)")
    parser.add_argument("csv", help="Input CSV with columns tic, sector, t0 (BTJD)")
    parser.add_argument("--out", help="Output CSV (default: <input>_pred.csv)")
    parser.add_argument("--plot", action="store_true", help="Show the prediction plot for each row")
    parser.add_argument("--no-contamination", action="store_true",
                        help="Skip the TIC contratio correction when computing radius_rjup")
    args = parser.parse_args()

    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_pred.csv")

    df = pd.read_csv(in_path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing column(s) in {in_path}: {', '.join(missing)}")

    for col in RESULT_COLUMNS:
        df[col] = None

    lc_cache = {}    # (tic, sector) -> light curve, so several t0 in one sector download once
    star_cache = {}  # tic -> stellar radius/contratio, shared with compute_param_single
    apply_contamination = not args.no_contamination

    for i, row in df.iterrows():
        tic, sector, t0 = int(row["tic"]), int(row["sector"]), float(row["t0"])
        print(f"\n[{i + 1}/{len(df)}] TIC {tic}  sector {sector}  t0 = {t0:.4f}")
        try:
            key = (tic, sector)
            if key not in lc_cache:
                lc_cache[key] = _download_sector(tic, sector)
            result = _run_inference_on_lc(lc_cache[key], t0, plot=args.plot)
        except Exception as exc:
            print(f"  ERROR -> {exc}. Skipping.")
            df.at[i, "error"] = str(exc)
            continue

        for col in RESULT_COLUMNS[:-3]:
            df.at[i, col] = result[col]
        radius_rjup = _planet_radius_rjup(tic, result["depth_ppm"], apply_contamination, star_cache)
        df.at[i, "radius_rjup"] = radius_rjup
        df.at[i, "radius_rearth"] = _rjup_to_rearth(radius_rjup)
        print(f"  depth = {result['depth_ppm']:.1f} ± {result['sigma_depth']:.1f} ppm   "
              f"duration = {result['duration_h']:.3f} ± {result['sigma_duration']:.3f} h   "
              f"t_mid = {result['t_mid_pred_btjd']:.6f} BTJD")
        if radius_rjup is not None:
            print(f"  radius = {radius_rjup:.4f} R_J = {_rjup_to_rearth(radius_rjup):.2f} R_E")

    df.to_csv(out_path, index=False)
    n_ok = df["error"].isna().sum()
    print(f"Done: {n_ok}/{len(df)} row(s) predicted. Results saved to {out_path}")


if __name__ == "__main__":
    main()
