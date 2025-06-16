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
import logging
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

import radiation_variation_analysis
import radiation_analysis
import fit_radiation_model
from operation_analysis import _plot_intensity_stats, _parse_rads
import dose_analysis
from bias_pipeline.bias_pipeline.steps.step4_generate_synthetic_bias import (
    generate_synthetic_bias,
)
from dark_pipeline.dark_pipeline.steps.step6_generate_synthetics import (
    generate_precise_synthetic_dark,
)

_STAGES: Iterable[str] = ("pre", "radiating", "post")


def _ensure_conversion(dataset_root: str) -> None:
    """Convert raw frames and build ``index.csv`` depending on dataset layout."""

    logger.info("Converting raw frames under %s", dataset_root)

    kdirs = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
        and _parse_rads(d) is not None
    ]

    if kdirs:
        rows: List[Dict[str, object]] = []
        rad_rows: List[Dict[str, float]] = []
        frame_num = 0
        prev_dose = 0.0
        for entry in sorted(kdirs, key=lambda d: _parse_rads(d) or 0.0):
            dpath = os.path.join(dataset_root, entry)
            fits_dir = os.path.join(dpath, "fits")
            if not os.path.isdir(fits_dir):
                continue
            files = sorted(glob.glob(os.path.join(fits_dir, "*.fits")))
            frame_nums: List[int] = []
            for fp in files:
                with fits.open(fp) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    hdr = hdul[0].header
                fr_num = hdr.get("FRAMENUM")
                if fr_num is None:
                    fr_num = frame_num
                    frame_num += 1
                else:
                    fr_num = int(fr_num)
                    frame_num = max(frame_num, fr_num + 1)
                frame_nums.append(fr_num)
                zero_frac = float(np.count_nonzero(data == 0)) / data.size
                rows.append(
                    {
                        "PATH": fp,
                        "CALTYPE": "DARK"
                        if float(hdr.get("EXPTIME", 0.0)) > 0.0
                        else "BIAS",
                        "STAGE": "radiating",
                        "VACUUM": None,
                        "TEMP": hdr.get("TEMP"),
                        "ZEROFRACTION": zero_frac,
                        "BADFITS": zero_frac > 0.01,
                    }
                )
            rad_csv = os.path.join(dpath, "radiationLogDef.csv")
            dose_val = _parse_rads(entry) or prev_dose
            if os.path.isfile(rad_csv):
                rdf = pd.read_csv(rad_csv)
                col = "Dose" if "Dose" in rdf.columns else "RadiationLevel"
                rads = pd.to_numeric(rdf[col], errors="coerce")
                fnums = (
                    pd.to_numeric(rdf.get("FrameNum"), errors="coerce")
                    if "FrameNum" in rdf.columns
                    else None
                )
                for idx, val in enumerate(rads):
                    if fnums is not None and pd.notna(fnums.iloc[idx]):
                        fn = int(fnums.iloc[idx])
                        frame_num = max(frame_num, fn + 1)
                    elif idx < len(frame_nums):
                        fn = frame_nums[idx]
                    else:
                        fn = frame_num
                        frame_num += 1
                    rad_rows.append({"FrameNum": fn, col: val})
                if rads.notna().any():
                    prev_dose = float(rads.iloc[-1])
            else:
                step = (dose_val - prev_dose) / len(frame_nums) if frame_nums else 0.0
                for fn in frame_nums:
                    prev_dose += step
                    rad_rows.append({"FrameNum": fn, "Dose": prev_dose})

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(dataset_root, "index.csv"), index=False)
        if rad_rows:
            pd.DataFrame(rad_rows).to_csv(
                os.path.join(dataset_root, "radiationLogCompleto.csv"), index=False
            )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            radiation_variation_analysis.main(dataset_root, tmp)


def _masters_to_npz(stage_dir: str) -> List[str]:
    """Convert master FITS frames in *stage_dir* to ``.npz`` archives."""
    npz_files: List[str] = []
    for path in glob.glob(os.path.join(stage_dir, "master_*.fits")):
        logger.debug("Converting %s", path)
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
    logger.info("Plotting intensity statistics for %s", stage_dir)

    for cal in ("bias", "dark"):
        csv_path = os.path.join(stage_dir, f"stats_{cal}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        times = df.get("FRAME", df.index).astype(float).tolist()
        means = df["MEAN"].astype(float).tolist()
        stds = df["STD"].astype(float).tolist()

        if cal == "dark" and "T_EXP" in df.columns:
            for exp, g in df.groupby("T_EXP"):
                t = g.get("FRAME", g.index).astype(float).tolist()
                m = g["MEAN"].astype(float).tolist()
                s = g["STD"].astype(float).tolist()
                fig_path = os.path.join(plots_dir, f"dark_intensity_{exp:.1f}s.png")
                _plot_intensity_stats(m, s, t, fig_path)
                np.savez_compressed(
                    os.path.join(plots_dir, f"dark_intensity_{exp:.1f}s.npz"),
                    time=t,
                    mean=m,
                    std=s,
                )
                pd.DataFrame({"TIME": t, "MEAN": m, "STD": s}).to_csv(
                    os.path.join(plots_dir, f"dark_intensity_{exp:.1f}s.csv"),
                    index=False,
                )

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
        logger.info("No frames found for radiation model in %s", stage_dir)
        return None
    logger.info("Fitting radiation model for %s", stage_dir)
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
        logger.info("Skipping reconstruction for %s (no params)", stage_dir)
        return

    recon_dir = os.path.join(stage_dir, "reconstruction")
    os.makedirs(recon_dir, exist_ok=True)

    coeffs: Dict[str, float] = {
        row["param"]: float(row["value"]) for _, row in params.iterrows()
    }

    for master_path in (p for p in (bias_master, dark_master) if p):
        logger.debug("Reconstructing %s", master_path)
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


def run_pipeline(
    dataset_root: str,
    radiation_log: str,
    output_dir: str,
    *,
    ignore_temp: bool = False,
) -> None:
    logger.info("Starting radiation pipeline")
    _ensure_conversion(dataset_root)
    index_csv = os.path.join(dataset_root, "index.csv")
    radiation_analysis.main(
        index_csv,
        radiation_log,
        output_dir,
        _STAGES,
        ignore_temp=ignore_temp,
    )

    for stage in _STAGES:
        stage_dir = os.path.join(output_dir, stage)
        if not os.path.isdir(stage_dir):
            continue
        logger.info("Processing stage %s", stage)
        _plot_stage_stats(stage_dir)
        _masters_to_npz(stage_dir)
        params = _fit_radiation_model(stage_dir)
        _reconstruct_and_compare(stage_dir, params)

    precision_dir = os.path.join(output_dir, "precision")
    os.makedirs(precision_dir, exist_ok=True)
    logger.info("Computing precision metrics")
    dose_analysis.main(index_csv, precision_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full radiation pipeline")
    parser.add_argument("dataset_root", help="Directory with Pre/Irradiation data")
    parser.add_argument("radiation_log", help="Path to radiationLogCompleto.csv")
    parser.add_argument("output_dir", help="Where to store results")
    parser.add_argument(
        "--ignore-temp",
        action="store_true",
        help="Do not group frames by temperature",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_pipeline(
        args.dataset_root,
        args.radiation_log,
        args.output_dir,
        ignore_temp=args.ignore_temp,
    )


if __name__ == "__main__":
    main()
