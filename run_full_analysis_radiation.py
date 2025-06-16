#!/usr/bin/env python3
"""Run a full analysis of irradiation FITS directories.

This script scans ``<dose>kRads`` folders containing ``fits/`` and
``radiationLogDef.csv`` files.  Every FITS is associated with a radiation
value following the same logic used by :mod:`run_full_radiation_pipeline`.
A per-frame CSV with the assigned dose, timestamp and estimated dose rate
is generated for downstream processing.  The standard radiation analysis
and dose analysis routines are executed afterwards and summary plots are
created showing the mean signal of bias and dark frames versus dose with
a ``fill_between`` shaded region for one standard deviation.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

import radiation_analysis
import dose_analysis
from run_full_radiation_pipeline import _ensure_conversion


logger = logging.getLogger(__name__)


def _build_frame_table(dataset_root: str) -> pd.DataFrame:
    """Return a DataFrame with dose and timestamp for each FITS file."""

    index_csv = os.path.join(dataset_root, "index.csv")
    rad_csv = os.path.join(dataset_root, "radiationLogCompleto.csv")
    idx_df = pd.read_csv(index_csv)
    rad_df = pd.read_csv(rad_csv)

    dose_col = "Dose" if "Dose" in rad_df.columns else "RadiationLevel"
    doses = pd.to_numeric(rad_df[dose_col], errors="coerce").tolist()
    frames = (
        pd.to_numeric(rad_df.get("FrameNum"), errors="coerce")
        if "FrameNum" in rad_df.columns
        else pd.Series(range(len(rad_df)))
    ).tolist()
    dose_map: Dict[int, float] = {int(f): float(d) for f, d in zip(frames, doses)}

    rows: List[dict[str, float]] = []
    for idx, row in idx_df.iterrows():
        path = row["PATH"]
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            fr = hdr.get("FRAMENUM")
            ts = hdr.get("TIMESTAMP")
            if ts is None:
                ts = hdr.get("TIME")
            exp = hdr.get("EXPTIME")
        if fr is None:
            fr = idx
        dose = dose_map.get(int(fr), doses[idx] if idx < len(doses) else np.nan)
        rows.append(
            {
                "PATH": path,
                "CALTYPE": row["CALTYPE"],
                "STAGE": row["STAGE"],
                "FRAME": int(fr),
                "TIMESTAMP": float(ts) if ts is not None else np.nan,
                "EXPTIME": float(exp) if exp is not None else np.nan,
                "DOSE": float(dose),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("TIMESTAMP")
    df["DOSE_RATE"] = df["DOSE"].diff() / df["TIMESTAMP"].diff()
    return df


def _plot_bias_mean(df: pd.DataFrame, out_png: str) -> None:
    sub = df[df["CALTYPE"] == "BIAS"].sort_values("DOSE")
    if sub.empty:
        return
    x = sub["DOSE"].astype(float)
    y = sub["MEAN"].astype(float)
    e = sub["STD"].astype(float)
    plt.figure()
    plt.plot(x, y, "-o", label="Mean")
    plt.fill_between(x, y - e, y + e, alpha=0.3, label="Std")
    plt.xlabel("Dose [kRad]")
    plt.ylabel("Mean ADU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_dark_mean(df: pd.DataFrame, out_dir: str) -> None:
    sub = df[df["CALTYPE"] == "DARK"]
    if sub.empty:
        return
    for exp in sorted(sub["EXPTIME"].dropna().unique()):
        s = sub[sub["EXPTIME"] == exp].sort_values("DOSE")
        x = s["DOSE"].astype(float)
        y = s["MEAN"].astype(float)
        e = s["STD"].astype(float)
        plt.figure()
        plt.plot(x, y, "-o", label="Mean")
        plt.fill_between(x, y - e, y + e, alpha=0.3, label="Std")
        plt.xlabel("Dose [kRad]")
        plt.ylabel("Mean ADU")
        plt.title(f"E={exp:g}s")
        plt.legend()
        plt.tight_layout()
        fname = f"dark_mean_vs_dose_E{str(exp).replace('.', 'p')}s.png"
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def run_pipeline(dataset_root: str, output_dir: str, *, ignore_temp: bool = False, verbose: bool = False) -> None:
    """Run the full irradiation analysis workflow."""

    logger.info("Preparing dataset under %s", dataset_root)
    _ensure_conversion(dataset_root)

    info_df = _build_frame_table(dataset_root)
    os.makedirs(output_dir, exist_ok=True)
    info_csv = os.path.join(output_dir, "frame_info.csv")
    info_df.to_csv(info_csv, index=False)
    logger.info("Frame information written to %s", info_csv)

    index_csv = os.path.join(dataset_root, "index.csv")
    rad_csv = os.path.join(dataset_root, "radiationLogCompleto.csv")

    stage_dir = os.path.join(output_dir, "radiation_analysis")
    radiation_analysis.main(index_csv, rad_csv, stage_dir, stages=["radiating"], ignore_temp=ignore_temp)

    dose_dir = os.path.join(output_dir, "dose_analysis")
    dose_analysis.main(index_csv, dose_dir, verbose)

    summary_csv = os.path.join(dose_dir, "dose_summary.csv")
    if os.path.isfile(summary_csv):
        summary = pd.read_csv(summary_csv)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        _plot_bias_mean(summary, os.path.join(plots_dir, "bias_mean_vs_dose.png"))
        _plot_dark_mean(summary, plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analysis on irradiation FITS")
    parser.add_argument("dataset_root", help="Directory containing <dose>kRads folders")
    parser.add_argument("output_dir", help="Directory for results")
    parser.add_argument("--ignore-temp", action="store_true", help="Do not group frames by temperature")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    run_pipeline(args.dataset_root, args.output_dir, ignore_temp=args.ignore_temp, verbose=args.verbose)


if __name__ == "__main__":
    main()
