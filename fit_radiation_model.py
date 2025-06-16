# -*- coding: utf-8 -*-
"""Fit radiation response model for bias and dark frames.

This script loads FITS or NumPy files from a directory, extracts the
median signal and variance per frame and fits the coefficients of the
following model (see user equations)::

    D* = B0 + alpha_D * D + [DC0 + beta_D * D] * t_exp + <q> * D_rate * t_exp
    sigma^2 = sigma_read^2 + [DC0 + beta_D * D] * t_exp
              + (<q>^2 + sigma_q^2) * D_rate * t_exp

Only the parameters ``B0``, ``alpha_D``, ``DC0``, ``beta_D``, ``<q>``,
``sigma_q`` and ``sigma_read`` are estimated.  Temperature terms are
ignored for simplicity.

Usage
-----
    python fit_radiation_model.py <frames_dir> <output_dir>

The directory must contain ``.fits`` or ``.npz`` files.  NumPy archives
are expected to store ``image_data`` and the metadata fields
``frame_type``, ``t_exp``, ``dose_rate``, ``dose_total`` and
``temperature``.
"""
from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Frame loading
# -----------------------------------------------------------------------------

def _load_fits(path: str) -> dict:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header
    return {
        "frame_type": str(hdr.get("FRAME_TYPE", hdr.get("IMAGETYP", ""))).lower(),
        "t_exp": float(hdr.get("EXPTIME", 0.0)),
        "dose_rate": float(hdr.get("DOSE_RATE", 0.0)),
        "dose_total": float(hdr.get("DOSE", 0.0)),
        "temperature": float(hdr.get("TEMP", np.nan)),
        "image_data": data,
    }


def _load_npz(path: str) -> dict:
    arr = np.load(path, allow_pickle=True)
    data = arr["image_data"]
    return {
        "frame_type": str(arr["frame_type"]),
        "t_exp": float(arr["t_exp"]),
        "dose_rate": float(arr["dose_rate"]),
        "dose_total": float(arr["dose_total"]),
        "temperature": float(arr["temperature"]),
        "image_data": data.astype(np.float32),
    }


def load_frames(directory: str) -> pd.DataFrame:
    """Return per-frame statistics from ``directory``."""
    rows = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            if name.lower().endswith(".fits"):
                meta = _load_fits(path)
            elif name.lower().endswith(".npz"):
                meta = _load_npz(path)
            else:
                continue
        except Exception:
            continue
        data = meta.pop("image_data")
        median = float(np.median(data))
        var = float(np.var(data, ddof=0))
        meta.update({"median": median, "variance": var})
        rows.append(meta)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Utility fitting functions
# -----------------------------------------------------------------------------

def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return coefficients (intercept, slope), 1-sigma errors, R2 and residuals.

    Any NaN or infinite values are removed before fitting. The function
    warns and returns ``NaN`` coefficients when the fit cannot be computed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        logger.debug("Filtered %d invalid points", np.sum(~mask))
    x = x[mask]
    y = y[mask]

    if x.size < 2 or np.allclose(x, x[0]):
        logger.warning("Degenerate data for linear fit")
        coeff = np.full(2, np.nan)
        errs = np.full(2, np.nan)
        return coeff, errs, float("nan"), np.full_like(y, np.nan)

    A = np.stack([np.ones_like(x), x], axis=1)
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        logger.warning("Linear fit failed: %s", exc)
        coeff = np.full(2, np.nan)
        errs = np.full(2, np.nan)
        return coeff, errs, float("nan"), np.full_like(y, np.nan)

    y_pred = A @ coeff
    resid = y - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    dof = max(0, len(x) - 2)
    sigma2 = ss_res / dof if dof > 0 else 0.0
    cov = sigma2 * np.linalg.inv(A.T @ A)
    errs = np.sqrt(np.diag(cov))
    return coeff, errs, r2, resid


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Approximate the inverse CDF of the standard normal distribution."""
    # Coefficients from Peter John Acklam's approximation
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [  7.784695709041462e-03,  3.224671290700398e-01,
           2.445134137142996e+00,  3.754408661907416e+00 ]
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = p - 0.5
    r = np.empty_like(p)
    mask = np.abs(q) <= 0.425
    if np.any(mask):
        s = 0.180625 - q[mask] * q[mask]
        r[mask] = q[mask] * (((((a[0]*s + a[1])*s + a[2])*s + a[3])*s + a[4])*s + a[5]) / \
            (((((b[0]*s + b[1])*s + b[2])*s + b[3])*s + b[4])*s + 1)
    mask = q > 0.425
    if np.any(mask):
        s = np.sqrt(-2*np.log(1-p[mask]))
        r[mask] = (((((c[0]*s + c[1])*s + c[2])*s + c[3])*s + c[4])*s + c[5]) / \
            ((((d[0]*s + d[1])*s + d[2])*s + d[3])*s + 1)
    mask = q < -0.425
    if np.any(mask):
        s = np.sqrt(-2*np.log(p[mask]))
        r[mask] = -(((((c[0]*s + c[1])*s + c[2])*s + c[3])*s + c[4])*s + c[5]) / \
            ((((d[0]*s + d[1])*s + d[2])*s + d[3])*s + 1)
    return r


# -----------------------------------------------------------------------------
# Model fitting
# -----------------------------------------------------------------------------

def fit_model(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Fit all model parameters and save diagnostics in *outdir*."""
    os.makedirs(outdir, exist_ok=True)
    df = df.sort_values("dose_total")

    # Classify stages
    first_irrad = df[df["dose_rate"] > 0]["dose_total"].min()
    last_irrad = df[df["dose_rate"] > 0]["dose_total"].max()
    base_mask = (df["dose_rate"] == 0) & (df["dose_total"] <= first_irrad)
    inbeam_mask = df["dose_rate"] > 0
    post_mask = (df["dose_rate"] == 0) & (df["dose_total"] >= last_irrad)

    # ------------------------------------------------------------------
    # Step 4.1: bias base level trend
    # ------------------------------------------------------------------
    bias_df = df[(df["frame_type"] == "bias") & (base_mask | post_mask)]
    B0, alpha_D, r2_b, res_b = np.nan, np.nan, np.nan, np.array([])
    if len(bias_df) >= 2:
        coeff, errs, r2_b, res = _linear_fit(bias_df["dose_total"].to_numpy(float),
                                             bias_df["median"].to_numpy(float))
        B0 = float(coeff[0])
        alpha_D = float(coeff[1])
        err_B0 = float(errs[0])
        err_alpha_D = float(errs[1])
        res_b = res
        # residual plot
        plt.figure()
        plt.scatter(bias_df["dose_total"], res)
        plt.axhline(0, color="k", ls="--")
        plt.xlabel("Dose")
        plt.ylabel("Residual")
        plt.title("Bias fit residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "bias_residuals.png"))
        plt.close()
    else:
        err_B0 = float("nan")
        err_alpha_D = float("nan")

    # ------------------------------------------------------------------
    # Step 4.2: dark current vs exposure and dose
    # ------------------------------------------------------------------
    dark_df = df[df["frame_type"] == "dark"]
    slopes = []
    doses = []
    for D, sub in dark_df.groupby("dose_total"):
        if sub["t_exp"].nunique() < 2:
            continue
        pred_bias = B0 + alpha_D * D
        y = sub["median"].to_numpy(float) - pred_bias
        x = sub["t_exp"].to_numpy(float)
        coeff, _, _, _ = _linear_fit(x, y)
        slopes.append(float(coeff[1]))
        doses.append(float(D))
    DC0, beta_D, r2_dc, res_dc = np.nan, np.nan, np.nan, np.array([])
    if len(doses) >= 2:
        coeff, errs, r2_dc, res_dc = _linear_fit(np.array(doses), np.array(slopes))
        DC0 = float(coeff[0])
        beta_D = float(coeff[1])
        err_DC0 = float(errs[0])
        err_beta_D = float(errs[1])
        plt.figure()
        plt.scatter(doses, res_dc)
        plt.axhline(0, color="k", ls="--")
        plt.xlabel("Dose")
        plt.ylabel("Residual")
        plt.title("Dark current fit residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "dark_current_residuals.png"))
        plt.close()
    else:
        err_DC0 = float("nan")
        err_beta_D = float("nan")

    # ------------------------------------------------------------------
    # Step 4.3: mean charge per particle
    # ------------------------------------------------------------------
    in_df = df[inbeam_mask & (df["t_exp"] > 0)]
    q_mean = float("nan")
    q_err = float("nan")
    if not in_df.empty:
        pred_bias = B0 + alpha_D * in_df["dose_total"].to_numpy(float)
        pred_dark = (DC0 + beta_D * in_df["dose_total"].to_numpy(float)) * in_df[
            "t_exp"].to_numpy(float)
        excess = in_df["median"].to_numpy(float) - pred_bias - pred_dark
        denom = in_df["dose_rate"].to_numpy(float) * in_df["t_exp"].to_numpy(float)
        valid = denom != 0
        q_vals = excess[valid] / denom[valid]
        if q_vals.size:
            q_mean = float(np.mean(q_vals))
            q_err = float(np.std(q_vals, ddof=1) / np.sqrt(q_vals.size))

    # ------------------------------------------------------------------
    # Step 4.4: noise parameters
    # ------------------------------------------------------------------
    pred_bias = B0 + alpha_D * df["dose_total"].to_numpy(float)
    pred_dark = (DC0 + beta_D * df["dose_total"].to_numpy(float)) * df[
        "t_exp"].to_numpy(float)
    q_rate = df["dose_rate"].to_numpy(float) * df["t_exp"].to_numpy(float)
    y = df["variance"].to_numpy(float) - pred_dark - (q_mean ** 2) * q_rate
    X = np.stack([np.ones(len(y)), q_rate], axis=1)
    coeff, errs, r2_noise, res_var = _linear_fit(X[:, 1], y)
    sigma_read = float(np.sqrt(max(coeff[0], 0.0)))
    sigma_q = float(np.sqrt(max(coeff[1], 0.0)))
    err_sigma_read = float(errs[0] / (2 * sigma_read)) if sigma_read > 0 else float("nan")
    err_sigma_q = float(errs[1] / (2 * sigma_q)) if sigma_q > 0 else float("nan")

    # QQ plot of final residuals
    n = len(res_var)
    prob = (np.arange(1, n + 1) - 0.5) / n
    norm_q = _norm_ppf(prob)
    sorted_res = np.sort(res_var)
    plt.figure()
    plt.plot(norm_q, sorted_res, "o")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Residual quantiles")
    plt.title("QQ plot")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "qq_plot.png"))
    plt.close()

    # Final residuals plot
    plt.figure()
    plt.scatter(pred_bias + pred_dark + q_mean * q_rate, res_var)
    plt.axhline(0, color="k", ls="--")
    plt.xlabel("Predicted mean")
    plt.ylabel("Variance residual")
    plt.title("Variance residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "variance_residuals.png"))
    plt.close()

    params = pd.DataFrame(
        [
            {"param": "B0", "value": B0, "err": err_B0, "R2": r2_b},
            {"param": "alpha_D", "value": alpha_D, "err": err_alpha_D, "R2": r2_b},
            {"param": "DC0", "value": DC0, "err": err_DC0, "R2": r2_dc},
            {"param": "beta_D", "value": beta_D, "err": err_beta_D, "R2": r2_dc},
            {"param": "q_mean", "value": q_mean, "err": q_err, "R2": np.nan},
            {"param": "sigma_q", "value": sigma_q, "err": err_sigma_q, "R2": r2_noise},
            {"param": "sigma_read", "value": sigma_read, "err": err_sigma_read, "R2": r2_noise},
        ]
    )

    params.to_csv(os.path.join(outdir, "fit_results.csv"), index=False)
    with open(os.path.join(outdir, "fit_results.json"), "w") as f:
        json.dump(params.to_dict(orient="records"), f, indent=2)

    return params


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

def main(frames_dir: str, output_dir: str) -> None:
    df = load_frames(frames_dir)
    if df.empty:
        print("No valid frames found")
        return
    result = fit_model(df, output_dir)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit radiation response model")
    parser.add_argument("frames_dir", help="Directory with FITS/NumPy files")
    parser.add_argument("output_dir", help="Where to store results")
    args = parser.parse_args()
    main(args.frames_dir, args.output_dir)
