#!/usr/bin/env python
"""Run the full radiation processing workflow.

This script ties together FITS conversion, dataset indexing,
per-stage analysis, radiation model fitting and photometric precision
assessment. It expects the standard irradiation dataset layout::

    <dataset_root>/Preirradiation/
    <dataset_root>/Irradiation/<dose>kRads/
    <dataset_root>/Postirradiation/

The required ``radiationLogCompleto.csv`` must be passed separately.
Results are grouped under ``output_dir`` in subfolders for each
irradiation stage (``pre``, ``radiating`` and ``post``).
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
from astropy.io import fits

import radiation_variation_analysis
import radiation_analysis
import fit_radiation_model
from operation_analysis import _plot_intensity_stats
import dose_analysis
from bias_pipeline.bias_pipeline.steps.step4_generate_synthetic_bias import (
    generate_synthetic_bias,
)
from dark_pipeline.dark_pipeline.steps.step6_generate_synthetics import (
    generate_precise_synthetic_dark,
)

_STAGES: Iterable[str] = ("pre", "radiating", "post")


def _ensure_conversion(dataset_root: str) -> None:
    """Run ``radiation_variation_analysis`` to convert raw frames and index."""
    with tempfile.TemporaryDirectory() as tmp:
        radiation_variation_analysis.main(dataset_root, tmp)


def _masters_to_npz(stage_dir: str) -> List[str]:
    """Convert master FITS frames in *stage_dir* to ``.npz`` archives."""
    npz_files: List[str] = []
    for path in glob.glob(os.path.join(stage_dir, "master_*.fits")):
        data = fits.getdata(path).astype(np.float32)
        hdr = fits.getheader(path)
        frame_type = "bias" if "bias" in os.path.basename(path).lower() else "dark"
        t_exp = float(hdr.get("EXPTIME", 0.0))
        dose_total = float(hdr.get("DOSE", 0.0))
        dose_rate = float(hdr.get("DOSE_RATE", 0.0))
        temp = hdr.get("TAVG")
        if temp is None:
            temp = hdr.get("TEMP")
        temperature = float(temp) if temp is not None else float("nan")

        out_path = os.path.splitext(path)[0] + ".npz"
        np.savez_compressed(
            out_path,
            image_data=data,
            frame_type=frame_type,
            t_exp=t_exp,
            dose_total=dose_total,
            dose_rate=dose_rate,
            temperature=temperature,
        )
        npz_files.append(out_path)
    return npz_files


def _plot_stage_stats(stage_dir: str) -> None:
    plots_dir = os.path.join(stage_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for cal in ("bias", "dark"):
        csv_path = os.path.join(stage_dir, f"stats_{cal}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        times = df.get("FRAME", df.index).astype(float).tolist()
        means = df["MEAN"].astype(float).tolist()
        stds = df["STD"].astype(float).tolist()
        fig_path = os.path.join(plots_dir, f"{cal}_intensity.png")
        _plot_intensity_stats(means, stds, times, fig_path)
        np.savez_compressed(
            os.path.join(plots_dir, f"{cal}_intensity.npz"),
            time=times,
            mean=means,
            std=stds,
        )
        pd.DataFrame({"TIME": times, "MEAN": means, "STD": stds}).to_csv(
            os.path.join(plots_dir, f"{cal}_intensity.csv"), index=False
        )


def _fit_radiation_model(stage_dir: str) -> pd.DataFrame | None:
    """Fit radiation model for all NPZ frames in *stage_dir*."""
    model_dir = os.path.join(stage_dir, "radiation_model")
    df = fit_radiation_model.load_frames(stage_dir)
    if df.empty:
        return None
    return fit_radiation_model.fit_model(df, model_dir)


def _reconstruct_and_compare(stage_dir: str, params: pd.DataFrame | None) -> None:
    """Generate synthetic frames from ``params`` and compare with masters."""
    bias_master = next(
        (f for f in glob.glob(os.path.join(stage_dir, "master_bias*.fits"))),
        None,
    )
    dark_master = next(
        (f for f in glob.glob(os.path.join(stage_dir, "master_dark*.fits"))),
        None,
    )
    if params is None:
        return

    recon_dir = os.path.join(stage_dir, "reconstruction")
    os.makedirs(recon_dir, exist_ok=True)

    coeffs: Dict[str, float] = {
        row["param"]: float(row["value"]) for _, row in params.iterrows()
    }

    for master_path in (p for p in (bias_master, dark_master) if p):
        arr = np.load(os.path.splitext(master_path)[0] + ".npz")
        data = arr["image_data"].astype(np.float32)
        ftype = str(arr["frame_type"])
        t_exp = float(arr["t_exp"])
        dose = float(arr["dose_total"])
        dose_rate = float(arr["dose_rate"])

        if ftype == "bias":
            pred = coeffs.get("B0", 0.0) + coeffs.get("alpha_D", 0.0) * dose
            synth = generate_synthetic_bias(np.full_like(data, pred), np.zeros_like(data), 0.0)
        else:
            pred = (
                coeffs.get("B0", 0.0)
                + coeffs.get("alpha_D", 0.0) * dose
                + (coeffs.get("DC0", 0.0) + coeffs.get("beta_D", 0.0) * dose) * t_exp
                + coeffs.get("q_mean", 0.0) * dose_rate * t_exp
            )
            synth = generate_precise_synthetic_dark(
                0.0,
                np.zeros_like(data),
                np.zeros_like(data),
                np.zeros_like(data, dtype=bool),
                pred,
                0.0,
                0.0,
                data.shape,
            )

        diff = synth - data
        diff_path = os.path.join(
            recon_dir, os.path.basename(master_path).replace(".fits", "_diff.fits")
        )
        fits.writeto(diff_path, diff.astype(np.float32), overwrite=True)
        png_path = diff_path.replace(".fits", ".png")
        radiation_analysis.diff_heatmap(data, synth, png_path, "Synthetic - master")
        np.savez_compressed(diff_path.replace(".fits", ".npz"), diff=diff)


def run_pipeline(dataset_root: str, radiation_log: str, output_dir: str) -> None:
    _ensure_conversion(dataset_root)
    index_csv = os.path.join(dataset_root, "index.csv")
    radiation_analysis.main(index_csv, radiation_log, output_dir, _STAGES)

    for stage in _STAGES:
        stage_dir = os.path.join(output_dir, stage)
        if not os.path.isdir(stage_dir):
            continue
        _plot_stage_stats(stage_dir)
        _masters_to_npz(stage_dir)
        params = _fit_radiation_model(stage_dir)
        _reconstruct_and_compare(stage_dir, params)

    precision_dir = os.path.join(output_dir, "precision")
    os.makedirs(precision_dir, exist_ok=True)
    dose_analysis.main(index_csv, precision_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full radiation pipeline")
    parser.add_argument("dataset_root", help="Directory with Pre/Irradiation data")
    parser.add_argument("radiation_log", help="Path to radiationLogCompleto.csv")
    parser.add_argument("output_dir", help="Where to store results")
    args = parser.parse_args()
    run_pipeline(args.dataset_root, args.radiation_log, args.output_dir)


if __name__ == "__main__":
    main()
