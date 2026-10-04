import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk

from computeparams.param_inference_core import (
    pick_lightcurve,
    preprocess_lc,
    _pick_t0_interactively,
    _run_inference_on_lc,
)
from computeparams.compute_radius import compute_planet_radius_from_tic, R_JUPITER_IN_R_SUN, R_EARTH_IN_R_SUN

SUMMARY_COLUMNS = ["sector", "transit", "depth_ppm", "duration_h", "t_mid_pred_btjd", "radius_rjup", "radius_rearth"]


def _planet_radius_rjup(tic_id, depth_ppm, apply_contamination, star_cache):
    
    if tic_id is None:
        return None

    depth_frac = depth_ppm / 1e6
    try:
        if tic_id not in star_cache:
            star_cache[tic_id] = compute_planet_radius_from_tic(
                tic_id, depth_frac, apply_contamination=apply_contamination, verbose=False,
            )
            return star_cache[tic_id]["Rp_Rjup"]

        cached = star_cache[tic_id]
        depth_used = depth_frac
        if cached["contamination_applied"]:
            depth_used = depth_frac * (1 + cached["contratio"])
        Rp_sun = cached["R_star_Rsun"] * (depth_used ** 0.5)
        return Rp_sun / R_JUPITER_IN_R_SUN
    except Exception as exc:
        print(f"  WARNING: could not compute planet radius for TIC {tic_id} -> {exc}")
        return None


def _rjup_to_rearth(radius_rjup):
    if radius_rjup is None:
        return None
    return radius_rjup * R_JUPITER_IN_R_SUN / R_EARTH_IN_R_SUN


PREFERRED_AUTHOR_ORDER = {"SPOC": 0, "TESS-SPOC": 1, "QLP": 2}

def _search_all_sectors(tic):

    sr = lk.search_lightcurve(f"TIC {tic}")
    if len(sr) == 0:
        return sr, {}

    by_sector = {}
    for i, row in enumerate(sr.table):
        sector, author = row["mission"], row["author"]
        rank = PREFERRED_AUTHOR_ORDER.get(author, 99)
        by_sector.setdefault(sector, []).append((rank, author, i))
    for options in by_sector.values():
        options.sort(key=lambda e: e[0])

    return sr, by_sector


def _resolve_sector_picks(by_sector, interactive=True):
    
    picks = {}
    for sector in sorted(by_sector):
        options = by_sector[sector]
        standard = [o for o in options if o[1] in PREFERRED_AUTHOR_ORDER] or options

        default_rank, default_author, default_idx = standard[0]
        
        seen = set()
        unique_options = []
        for _, author, idx in standard:
            if author not in seen:
                seen.add(author)
                unique_options.append((author, idx))

        if interactive and len(unique_options) > 1:
            menu = "\n".join(
                f"    [{n}] {author}" + ("  (default)" if author == default_author else "")
                for n, (author, _) in enumerate(unique_options, start=1)
            )
            choice = input(
                f"  sector {sector}: multiple pipelines available —\n{menu}\n"
                f"  Enter a number (Enter=default): "
            ).strip()
            if choice:
                try:
                    default_author, default_idx = unique_options[int(choice) - 1]
                except (ValueError, IndexError):
                    print(f"    invalid choice '{choice}', using {default_author}.")

        picks[sector] = (default_author, default_idx)

    return picks


def _download_picked_sectors(tic, search_result, picks):
    
    print(f"Downloading {len(picks)} sector(s) for TIC {tic}:")
    lcs_by_sector = {}
    for sector, (author, idx) in sorted(picks.items()):
        print(f"  sector {sector}  author={author}")
        try:
            lc = search_result[idx].download()
            lc = pick_lightcurve(lc)
            lc = preprocess_lc(lc)
        except Exception as exc:
            print(f"  WARNING: failed to download/read sector {sector} ({author}) -> {exc}. Skipping.")
            continue
        lcs_by_sector[lc.meta.get("SECTOR", sector)] = lc

    return lcs_by_sector


def _pick_transits_on_overview(tic, lcs_by_sector):
    sectors = sorted(lcs_by_sector)
    n = len(sectors)

    # Beyond 6 sectors a single column runs off the screen, fix..
    ncols = 1 if n <= 6 else (2 if n <= 12 else 3)
    nrows = math.ceil(n / ncols)
    panel_h = 2.6 if ncols > 1 else 3.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, panel_h * nrows),
                              constrained_layout=True, squeeze=False)
    flat_axes = axes.flatten()

    clicks = {sector: [] for sector in sectors}
    ax_to_sector = {}

    for ax, sector in zip(flat_axes, sectors):
        lc = lcs_by_sector[sector]
        t = np.asarray(lc.time.value, dtype=float)
        f = np.asarray(lc.flux.value, dtype=float)
        f_norm = f / np.nanmedian(f)
        ax.plot(t, f_norm, lw=0.6, color="steelblue")
        ax.set_title(f"sector {sector} — click each transit center", fontsize=9)
        ax.set_ylabel("norm. flux")
        ax.set_xlabel("BTJD − 2457000")
        ax.grid(True, alpha=0.2)
        ax_to_sector[ax] = sector

    for ax in flat_axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"TIC {tic} — click every transit center per sector, close window to confirm")

    def on_click(event):
        sector = ax_to_sector.get(event.inaxes)
        if sector is None or event.xdata is None:
            return
        clicks[sector].append(event.xdata)
        event.inaxes.axvline(event.xdata, color="red", lw=1.3, ls="--")
        event.inaxes.set_title(f"sector {sector} — {len(clicks[sector])} transit(s) marked")
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    return {sector: t0s for sector, t0s in clicks.items() if t0s}


def _run_for_sector(lc, t0_arg, sector_label=None, tic_id=None,
                     star_cache=None, apply_contamination=True, transit_label=1):
    
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    f_norm = f / np.nanmedian(f)

    if t0_arg is not None:
        t0_hat = t0_arg
    else:
        print(f"TIME range: {t.min():.4f} – {t.max():.4f} (BTJD)")
        t0_hat = _pick_t0_interactively(t, f_norm)
        if t0_hat is None:
            print("No point selected — skipping this sector.")
            return None

    if sector_label is None:
        sector_label = lc.meta.get("SECTOR")

    print(f"t0_hat = {t0_hat:.4f} BTJD")
    result = _run_inference_on_lc(lc, t0_hat, plot=True)

    radius_rjup = _planet_radius_rjup(
        tic_id, result["depth_ppm"], apply_contamination, star_cache if star_cache is not None else {},
    )

    return {
        "sector":          sector_label,
        "transit":         transit_label,
        "t0_hat_btjd":     t0_hat,
        "depth_ppm":       result["depth_ppm"],
        "sigma_depth":     result["sigma_depth"],
        "duration_h":      result["duration_h"],
        "sigma_duration":  result["sigma_duration"],
        "dtcorr_h":        result["dtcorr_h"],
        "sigma_dtcorr":    result["sigma_dtcorr"],
        "t_mid_pred_btjd": result["t_mid_pred_btjd"],
        "radius_rjup":     radius_rjup,
        "radius_rearth":   _rjup_to_rearth(radius_rjup),
    }


def _print_summary(tic_label, rows):
    if not rows:
        print("No sector produced a prediction — nothing to summarize.")
        return

    display_rows = []
    for row in rows:
        radius = f"{row['radius_rjup']:.4f}" if row["radius_rjup"] is not None else "n/a"
        radius_e = f"{row['radius_rearth']:.2f}" if row["radius_rearth"] is not None else "n/a"
        display_rows.append({
            "sector":          row["sector"],
            "transit":         row["transit"],
            "depth_ppm":       f"{row['depth_ppm']:.1f} ± {row['sigma_depth']:.1f}",
            "duration_h":      f"{row['duration_h']:.3f} ± {row['sigma_duration']:.3f}",
            "t_mid_pred_btjd": f"{row['t_mid_pred_btjd']:.6f} ± {row['sigma_dtcorr'] / 24.0:.6f}",
            "radius_rjup":     radius,
            "radius_rearth":   radius_e,
        })

    df = pd.DataFrame(display_rows, columns=SUMMARY_COLUMNS)
    print(f"=== Summary for {tic_label} ({len(df)} sector(s)) ===")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Single-transit parameter inference for one object")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file",   help="Path or URL to FITS light curve")
    src.add_argument("--tic",    type=int, help="TIC ID")

    parser.add_argument("--sector", type=int, nargs="+",
                        help="TESS sector(s) to run (used with --tic). "
                             "Omit to first browse every available sector.")
    parser.add_argument("--t0",     type=float,
                        help="Transit epoch in BTJD — omit to click interactively "
                             "(applied to every selected sector)")
    parser.add_argument("--no-contamination", action="store_true",
                        help="Skip the TIC contratio correction when computing radius_rjup "
                             "(default: apply it, same as compute_radius.py)")
    args = parser.parse_args()
    apply_contamination = not args.no_contamination

    if args.tic is not None:
        summary_rows = []
        star_cache = {}

        if args.sector:
            for sector in args.sector:
                print(f"Searching TIC {args.tic} sector {sector}…")
                sr = lk.search_lightcurve(f"TIC {args.tic}", sector=sector, author="SPOC")
                if len(sr) == 0:
                    print(f"  No SPOC light curve found for sector {sector}, skipping.")
                    continue
                lc = pick_lightcurve(sr[0].download())
                lc = preprocess_lc(lc)
                row = _run_for_sector(lc, args.t0, sector_label=sector, tic_id=args.tic,
                                       star_cache=star_cache, apply_contamination=apply_contamination)
                if row is not None:
                    summary_rows.append(row)
            _print_summary(f"TIC {args.tic}", summary_rows)
            return

        print(f"No --sector given, browsing every available sector for TIC {args.tic}…")
        sr, by_sector = _search_all_sectors(args.tic)
        if len(sr) == 0:
            print("No light curve found for this TIC.")
            return
        sector_picks = _resolve_sector_picks(by_sector)
        lcs_by_sector = _download_picked_sectors(args.tic, sr, sector_picks)
        if not lcs_by_sector:
            print("No sector could be downloaded/read.")
            return

        print("Click each transit center directly on the plot (one or more per sector — "
              "e.g. for a multi-planet system), close the window when done.")
        transit_picks = _pick_transits_on_overview(args.tic, lcs_by_sector)
        if not transit_picks:
            print("No transit marked — exiting.")
            return

        for sector in sorted(transit_picks):
            for transit_idx, t0_hat in enumerate(transit_picks[sector], start=1):
                row = _run_for_sector(lcs_by_sector[sector], t0_hat, sector_label=sector, tic_id=args.tic,
                                       star_cache=star_cache, apply_contamination=apply_contamination,
                                       transit_label=transit_idx)
                if row is not None:
                    summary_rows.append(row)
        _print_summary(f"TIC {args.tic}", summary_rows)
        return

    lc = lk.read(args.file)
    lc = pick_lightcurve(lc)
    lc = preprocess_lc(lc)
    row = _run_for_sector(lc, args.t0)
    _print_summary(args.file, [row] if row is not None else [])


if __name__ == "__main__":
    main()
