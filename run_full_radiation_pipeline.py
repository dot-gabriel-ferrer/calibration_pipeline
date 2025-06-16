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
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.io import fits

import radiation_variation_analysis
import radiation_analysis
import fit_radiation_model
from operation_analysis import _plot_intensity_stats
import dose_analysis

_STAGES: Iterable[str] = ("pre", "radiating", "post")


def _ensure_conversion(dataset_root: str) -> None:
    """Run ``radiation_variation_analysis`` to convert raw frames and index."""
    with tempfile.TemporaryDirectory() as tmp:
        radiation_variation_analysis.main(dataset_root, tmp)


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


def _fit_radiation_model(stage_dir: str) -> None:
    model_dir = os.path.join(stage_dir, "radiation_model")
    fit_radiation_model.main(stage_dir, model_dir)


def _reconstruct_and_compare(stage_dir: str) -> None:
    """Generate synthetic bias/dark and compare with masters if possible."""
    bias_master = next(
        (f for f in glob.glob(os.path.join(stage_dir, "master_bias*.fits"))),
        None,
    )
    dark_master = next(
        (f for f in glob.glob(os.path.join(stage_dir, "master_dark*.fits"))),
        None,
    )
    model_dir = os.path.join(stage_dir, "radiation_model")
    a_map = os.path.join(model_dir, "A_map.fits")
    b_map = os.path.join(model_dir, "B_map.fits")
    if bias_master and os.path.isfile(a_map) and os.path.isfile(b_map):
        temp = fits.getheader(bias_master).get("TAVG")
        if temp is None:
            temp = fits.getheader(bias_master).get("TEMP")
        out_path = os.path.join(model_dir, "synthetic_bias.fits")
        os.system(
            f"python bias_pipeline/bias_pipeline/generate_bias.py --a-map {a_map} --b-map {b_map} --temperature {temp} --output {out_path}"
        )
        if os.path.isfile(out_path):
            synth = fits.getdata(out_path)
            master = fits.getdata(bias_master)
            diff = synth - master
            fits.writeto(
                os.path.join(model_dir, "bias_diff.fits"),
                diff.astype(np.float32),
                overwrite=True,
            )

    dark_model = os.path.join(model_dir, "synthetic_dark.fits")
    if dark_master and os.path.isfile(dark_model):
        synth = fits.getdata(dark_model)
        master = fits.getdata(dark_master)
        diff = synth - master
        fits.writeto(
            os.path.join(model_dir, "dark_diff.fits"),
            diff.astype(np.float32),
            overwrite=True,
        )


def run_pipeline(dataset_root: str, radiation_log: str, output_dir: str) -> None:
    _ensure_conversion(dataset_root)
    index_csv = os.path.join(dataset_root, "index.csv")
    radiation_analysis.main(index_csv, radiation_log, output_dir, _STAGES)

    for stage in _STAGES:
        stage_dir = os.path.join(output_dir, stage)
        if not os.path.isdir(stage_dir):
            continue
        _plot_stage_stats(stage_dir)
        _fit_radiation_model(stage_dir)
        _reconstruct_and_compare(stage_dir)

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
