import argparse
import csv
import math
from pathlib import Path

import pandas as pd
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

from core.config import cfg

R_JUPITER_IN_R_SUN = 0.1028    # 1 R_J = 0.1028 R_sun
R_EARTH_IN_R_SUN   = 0.009158  # 1 R_E = 0.009158 R_sun


def get_radius_from_gaia(ra, dec, radius_arcsec=3.0):

    query = f"""
        SELECT TOP 1 ap.radius_gspphot
        FROM gaiadr3.gaia_source AS gs
        JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
        WHERE 1 = CONTAINS(
            POINT('ICRS', gs.ra, gs.dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600.0})
        )
        ORDER BY gs.phot_g_mean_mag ASC
    """
    job = Gaia.launch_job(query)
    result = job.get_results()
    if len(result) == 0 or result["radius_gspphot"].mask[0]:
        return None
    return float(result["radius_gspphot"][0])


def compute_planet_radius_from_tic(tic_id, transit_depth, apply_contamination=False, verbose=True):

    target_observations = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id)).to_pandas()
    if target_observations.empty:
        raise ValueError(f"No TIC entry found for TIC {tic_id}")

    tic_row = target_observations.iloc[0]
    R_star = tic_row["rad"]
    used_gaia_fallback = False

    if pd.isna(R_star):
        print(f"  TIC {tic_id}: no radius in TIC, trying Gaia...")
        R_star = get_radius_from_gaia(tic_row["ra"], tic_row["dec"])
        if R_star is None:
            raise ValueError(f"TIC {tic_id} has no stellar radius in TIC nor Gaia DR3")
        used_gaia_fallback = True

    depth = transit_depth
    contratio = tic_row.get("contratio", 0.0)
    contamination_applied = False
    if apply_contamination and pd.notna(contratio):
        depth = transit_depth * (1 + contratio)
        contamination_applied = True

    tess_mag = tic_row.get("Tmag")  
    Rp_sun = R_star * math.sqrt(depth)
    Rp_rj = Rp_sun / R_JUPITER_IN_R_SUN

    if verbose:
        contam_msg = f"yes (contratio={contratio:.4f})" if contamination_applied else "no"
        source_msg = "Gaia DR3 fallback" if used_gaia_fallback else "TIC catalog"
        print(f"TIC {tic_id}: R_star={R_star:.3f} R_sun (source: {source_msg}), "
              f"contamination applied: {contam_msg}, depth_used={depth:.6f}, "
              f"Rp={Rp_sun:.3f} R_sun = {Rp_rj:.4f} R_J")
        if R_star > 10:
            print("Warning: host star is a giant (R_star > 10 R_sun).")
        if Rp_rj > 2:
            print("Warning: companion radius > 2 R_J (likely non-planet).")

    return {
        "tic_id": tic_id,
        "R_star_Rsun": R_star,
        "transit_depth": transit_depth,
        "depth_used": depth,
        "Rp_Rsun": Rp_sun,
        "Rp_Rjup": Rp_rj,
        "contamination_applied": contamination_applied,
        "contratio": contratio,
        "used_gaia_fallback": used_gaia_fallback,
        "tess_mag": tess_mag,   
    }


def same_tic(a, b):
    return int(float(str(a).strip())) == int(float(str(b).strip()))


def load_rows(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "final_radius" not in fieldnames:
        fieldnames.append("final_radius")
    if "stellar_radius" not in fieldnames:         
        fieldnames.append("stellar_radius")
    if "tess_mag" not in fieldnames:               
        fieldnames.append("tess_mag")
    for row in rows:
        row.setdefault("final_radius", "")
        row.setdefault("stellar_radius", "")        
        row.setdefault("tess_mag", "")             
    return fieldnames, rows


def save_rows(csv_path, fieldnames, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fill_final_radius(csv_path, apply_contamination=True, tic_filter=None):
    fieldnames, rows = load_rows(csv_path)
    tic_cache = {}
    updated_count = 0

    for row in rows:
        tic_id = str(row.get("tic_id", "")).strip()
        if not tic_id:
            continue
        if tic_filter is not None and not same_tic(tic_id, tic_filter):
            continue

        depth_pred_raw = str(row.get("depth_pred", "")).strip()
        final_radius_raw = str(row.get("final_radius", "")).strip().lower()
        stellar_radius_raw = str(row.get("stellar_radius", "")).strip().lower()
        tess_mag_raw = str(row.get("tess_mag", "")).strip().lower()

        if final_radius_raw == "nan":
            final_radius_raw = ""
        if stellar_radius_raw == "nan":
            stellar_radius_raw = ""
        if tess_mag_raw == "nan":
            tess_mag_raw = ""

        already_has_radius = bool(final_radius_raw)

        if not depth_pred_raw or (final_radius_raw and stellar_radius_raw and tess_mag_raw):
            continue

        depth_frac = float(depth_pred_raw) / 1e6  # in ppm

        try:
            if tic_id not in tic_cache:
                tic_cache[tic_id] = compute_planet_radius_from_tic(
                    tic_id, depth_frac, apply_contamination=apply_contamination, verbose=False,
                )
                result = tic_cache[tic_id]
            else:
                cached = tic_cache[tic_id]
                if pd.isna(cached["R_star_Rsun"]):
                    raise ValueError(f"TIC {tic_id} has no stellar radius ('rad') in the TIC catalog")
                depth_used = depth_frac
                if cached["contamination_applied"]:
                    depth_used = depth_frac * (1 + cached["contratio"])
                Rp_sun = cached["R_star_Rsun"] * (depth_used ** 0.5)
                result = {
                    "Rp_Rjup": Rp_sun / R_JUPITER_IN_R_SUN,
                    "R_star_Rsun": cached["R_star_Rsun"],
                    "depth_used": depth_used,
                    "contamination_applied": cached["contamination_applied"],
                    "contratio": cached["contratio"],
                    "tess_mag": cached["tess_mag"],   
                }
        except Exception as exc:
            print(f"  TIC {tic_id}: could not compute final_radius -> {exc}")
            continue

        if pd.isna(result["Rp_Rjup"]):
            print(f"  TIC {tic_id}: result is NaN, leaving final_radius blank")
            continue

        row["stellar_radius"] = f"{result['R_star_Rsun']:.4f}"   
        if pd.notna(result.get("tess_mag")):
            row["tess_mag"] = f"{result['tess_mag']:.4f}"        

        if already_has_radius:
            
            updated_count += 1
            print(f"  TIC {tic_id}: final_radius already set ({row['final_radius']} R_J, untouched), "
                  f"stellar_radius backfilled = {row['stellar_radius']} R_sun, "
                  f"tess_mag backfilled = {row.get('tess_mag', '')}")
            continue

        row["final_radius"] = f"{result['Rp_Rjup']:.4f}"
        updated_count += 1
        contam_msg = f"yes (contratio={result['contratio']:.4f})" if result["contamination_applied"] else "no"
        print(
            f"  TIC {tic_id}: R_star={result['R_star_Rsun']:.3f} R_sun, "
            f"contamination applied: {contam_msg}, depth_used={result['depth_used']:.6f}, "
            f"final_radius = {row['final_radius']} R_J"
        )

    if updated_count:
        save_rows(csv_path, fieldnames, rows)
    print(f"Done. {updated_count} row(s) got final_radius filled in {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill final_radius (R_J) from depth_pred for a candidates CSV")
    parser.add_argument("--csv", type=Path, default=Path(cfg["paths"]["radius_csv"]),
                        help="Candidates CSV to fill in-place (default: paths.radius_csv in core/config.yaml)")
    parser.add_argument("--tic", type=int, default=None, help="Only fill this TIC (default: all rows)")
    parser.add_argument("--no-contamination", action="store_true",
                         help="Skip the TIC contratio correction (default: apply it)")
    args = parser.parse_args()

    fill_final_radius(args.csv, apply_contamination=not args.no_contamination, tic_filter=args.tic)