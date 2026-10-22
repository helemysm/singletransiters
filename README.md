# Single-Transit Parameter Estimation with Calibrated Uncertainties

A 1D convolutional neural network that estimates the parameters of a **single transit** in a TESS light curve directly from the observed flux, without detrending, phase-folding, or stellar priors.

Single transits cannot be phase-folded, and classical fits on un-detrended light curves often lock onto stellar variability instead of the transit. This network is trained on synthetic light curves that include red noise, quasi-periodic stellar variability, and instrumental outliers, and learns to separate the transit from these effects. It only needs an approximate transit time: it reads a ~100 h window around it and corrects the transit center by itself.

Given a TIC ID (or a FITS light curve) and the approximate transit time, it returns:

- transit depth (ppm)
- transit duration (hours)
- corrected mid-transit time (BTJD)
- planet radius in Jupiter and Earth radii, from the stellar radius in the TIC (or Gaia DR3 if the TIC has none)


## Installation

Use versions listed on requirements.txt

Light curves are downloaded from MAST with `lightkurve`, and stellar parameters are queried from the TIC (MAST) and Gaia DR3. 


### Example: one TIC (interactive)

```bash
python -m vetting.compute_param_single --tic 466206508
```

This example uses TOI-5542 b (TIC 466206508), a single transit in **sector 13**.

1. The script lists every TESS sector available for the TIC. When a sector has more than one pipeline (SPOC, TESS-SPOC, QLP), it asks which one to use. Press Enter to keep the default (priority: SPOC > TESS-SPOC > QLP).
2. A window opens with one panel per sector. In the **sector 13** panel, click the center of the transit, near **BTJD ≈ 1679.4**. Leave the other panels without clicks; they are skipped. Close the window to continue.
3. A plot of the prediction opens (start, center and end of the transit). Close it.
4. A summary table is printed:

```
        sector  transit       depth_ppm    duration_h        t_mid_pred_btjd radius_rjup radius_rearth
TESS Sector 13        1 11023.1 ± 452.4 9.139 ± 0.315 1679.356410 ± 0.002753      1.3154         14.77
```

You can click more than one transit, in one or several sectors. Each click gives one row in the summary.

Other options (just as examples):

```bash
# Only some sectors (SPOC light curves only)
python -m vetting.compute_param_single --tic 466206508 --sector 13

# Give the transit center instead of clicking
python -m vetting.compute_param_single --tic 466206508 --sector 13 --t0 ...

# If you have a local FITS light curve, use it
python -m vetting.compute_param_single --file path/to/lc.fits --t0 1691.24
```

| Option | Description |
|---|---|
| `--tic` | TIC ID |
| `--file` | Path or URL to a FITS light curve (instead of `--tic`) |
| `--sector` | One or more sectors. Only SPOC light curves are searched; without it, every pipeline is considered |
| `--t0` | Approximate transit center in BTJD (BJD − 2457000). Skips the click |
| `--no-contamination` | Do not correct the depth for flux contamination (TIC `contratio`) when computing the radius |

### If you have a list of TICs (CSV, no see as an interactive way)

Write a CSV with one transit per row:

```csv
tic,sector,t0
333736132,36,2283.58
209464063,4,1424.71
56815340,46,2569.43
```

`t0` is the approximate transit center in BTJD. 

```bash
python -m vetting.predict_csv my_targets.csv
python -m vetting.predict_csv my_targets.csv --out results.csv --plot
```


## Configuration

`core/config.yaml` holds the model to load:

```yaml
model_class: SmallConvUncRegressor
paths:
  model: "models/baseline_conv1d.pt"
  aux: "models/baseline_norm_calib.npz"
```
