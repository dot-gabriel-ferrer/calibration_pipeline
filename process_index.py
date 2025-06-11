#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Index-based calibration processing and analysis.

This script takes the CSV produced by ``utils.index_dataset`` and generates
master calibration frames (bias, dark and flat), computes per-frame statistics
and produces basic plots to evaluate detector behaviour.  It is intended as a
simple automation helper for quick analysis of radiation test datasets.

Usage
-----

```
python process_index.py path/to/index.csv output_dir/
```

The output directory will contain:

* ``masters/`` – master biases grouped by temperature and master dark/flat
  frames grouped by temperature and exposure time.
* ``frame_stats.csv`` – per frame statistics (mean, median, standard deviation
  and percentiles).
* ``plots/`` – trend plots for bias, dark and flat frames.
* ``comparisons/`` – visual comparison between each master frame and the most
  deviant individual frame used to build it.
"""

from __future__ import annotations

import os
import argparse
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

from utils.raw_to_fits import parse_filename_metadata


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_fits(path: str) -> np.ndarray:
    """Load FITS data as float32."""
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)


def _parse_temp_exp_from_path(path: str) -> tuple[float | None, float | None]:
    """Return (temperature, exptime) parsed from any component of *path*."""
    temp = None
    exp = None
    for part in path.split(os.sep):
        e, t = parse_filename_metadata(part)
        if t is not None and temp is None:
            try:
                temp = float(t)
            except ValueError:
                pass
        if e is not None and exp is None:
            try:
                exp = float(e)
            except ValueError:
                pass
        if temp is None:
            m = re.search(r"[Tt](-?[0-9]+(?:\.[0-9]+)?)", part)
            if m:
                try:
                    temp = float(m.group(1))
                except ValueError:
                    pass
        if exp is None:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)s", part)
            if m:
                try:
                    exp = float(m.group(1))
                except ValueError:
                    pass
    return temp, exp


def _make_mean_master(paths: list[str], temps: list[float] | None = None, exps: list[float] | None = None) -> tuple[np.ndarray, fits.Header]:
    """Compute mean master image and header with stats."""
    stack = np.stack([_load_fits(p) for p in paths], axis=0)
    master = np.mean(stack, axis=0)

    hdr = fits.Header()
    hdr["NSOURCE"] = len(paths)
    hdr["MEAN"] = float(np.mean(stack))
    hdr["MEDIAN"] = float(np.median(stack))
    hdr["STD"] = float(np.std(stack))
    hdr["DATAMIN"] = float(np.min(stack))
    hdr["DATAMAX"] = float(np.max(stack))
    if temps:
        hdr["TMIN"] = float(np.min(temps))
        hdr["TMAX"] = float(np.max(temps))
        hdr["TAVG"] = float(np.mean(temps))
    if exps:
        hdr["EMIN"] = float(np.min(exps))
        hdr["EMAX"] = float(np.max(exps))
        hdr["EAVG"] = float(np.mean(exps))
    return master, hdr


def load_index(index_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ``index.csv`` and return data frames for each calibration type."""
    df = pd.read_csv(index_path)
    df = df[df["BADFITS"] == False]  # ignore bad files
    bias_df = df[df["CALTYPE"] == "BIAS"].copy()
    dark_df = df[df["CALTYPE"] == "DARK"].copy()
    flat_df = df[df["CALTYPE"] == "FLAT"].copy()
    return bias_df, dark_df, flat_df


def _attempt_from_path(path: str) -> str:
    """Infer the attempt directory from a FITS path."""
    # expected .../attemptX/fits/file.fits -> return .../attemptX
    parent = os.path.dirname(path)
    attempt = os.path.basename(os.path.dirname(parent))
    return attempt


def master_bias_by_temp(bias_df: pd.DataFrame, outdir: str) -> dict[float, np.ndarray]:
    """Generate master bias frames per attempt and per temperature."""
    os.makedirs(outdir, exist_ok=True)
    temp_attempt_groups: dict[tuple[float, str], list[dict]] = defaultdict(list)

    for _, row in tqdm(
        bias_df.iterrows(), total=len(bias_df), desc="Grouping bias frames"
    ):
        eq_temp, _ = _parse_temp_exp_from_path(row["PATH"])
        temp = eq_temp if eq_temp is not None else row["TEMP"]
        attempt = _attempt_from_path(row["PATH"])
        temp_attempt_groups[(temp, attempt)].append(
            {"path": row["PATH"], "temp": row["TEMP"]}
        )

    temp_to_attempt_master: dict[float, list[np.ndarray]] = defaultdict(list)
    temp_to_all_temps: dict[float, list[float]] = defaultdict(list)

    for (temp, attempt), entries in tqdm(
        temp_attempt_groups.items(), desc="Combining bias per attempt"
    ):
        paths = [e["path"] for e in entries]
        temps = [e["temp"] for e in entries]
        master, hdr = _make_mean_master(paths, temps=temps)
        out_name = f"master_bias_{attempt}_T{temp:.1f}.fits"
        fits.writeto(os.path.join(outdir, out_name), master.astype(np.float32), hdr, overwrite=True)
        temp_to_attempt_master[temp].append(master)
        temp_to_all_temps[temp].extend(temps)

    master_per_temp: dict[float, np.ndarray] = {}
    for temp, masters in tqdm(
        temp_to_attempt_master.items(), desc="Writing master per temp"
    ):
        stack = np.stack(masters, axis=0)
        temps = temp_to_all_temps[temp]
        mtemp = np.mean(stack, axis=0)
        hdr = fits.Header()
        hdr["NSOURCE"] = stack.shape[0]
        hdr["MEAN"] = float(np.mean(stack))
        hdr["MEDIAN"] = float(np.median(stack))
        hdr["STD"] = float(np.std(stack))
        hdr["DATAMIN"] = float(np.min(stack))
        hdr["DATAMAX"] = float(np.max(stack))
        if temps:
            hdr["TMIN"] = float(np.min(temps))
            hdr["TMAX"] = float(np.max(temps))
            hdr["TAVG"] = float(np.mean(temps))
        out_name = f"master_bias_T{temp:.1f}.fits"
        fits.writeto(os.path.join(outdir, out_name), mtemp.astype(np.float32), hdr, overwrite=True)
        master_per_temp[temp] = mtemp
    return master_per_temp


def master_dark_flat(
    dark_df: pd.DataFrame,
    flat_df: pd.DataFrame,
    outdir_dark: str,
    outdir_flat: str,
) -> tuple[dict[tuple[float, float], np.ndarray], dict[tuple[float, float], np.ndarray]]:
    """Generate master darks and flats grouped by temperature and exposure."""
    os.makedirs(outdir_dark, exist_ok=True)
    os.makedirs(outdir_flat, exist_ok=True)

    dark_groups: dict[tuple[float, float, str], list[dict]] = defaultdict(list)

    for _, row in tqdm(dark_df.iterrows(), total=len(dark_df), desc="Grouping dark frames"):
        eq_temp, p_exp = _parse_temp_exp_from_path(row["PATH"])
        hdr = fits.getheader(row["PATH"])
        exp = p_exp if p_exp is not None else hdr.get("EXPTIME")
        temp = eq_temp if eq_temp is not None else row["TEMP"]
        attempt = _attempt_from_path(row["PATH"])
        dark_groups[(temp, exp, attempt)].append({"path": row["PATH"], "temp": row["TEMP"], "exp": exp})

    per_group_masters: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    per_group_temps: dict[tuple[float, float], list[float]] = defaultdict(list)

    for (t, e, attempt), entries in tqdm(dark_groups.items(), desc="Combining dark per attempt"):
        paths = [e["path"] for e in entries]
        temps = [e["temp"] for e in entries]
        exps = [e["exp"] for e in entries]
        master, hdr = _make_mean_master(paths, temps=temps, exps=exps)
        name = f"master_dark_{attempt}_T{t:.1f}_E{e:.1f}.fits"
        fits.writeto(os.path.join(outdir_dark, name), master.astype(np.float32), hdr, overwrite=True)
        per_group_masters[(t, e)].append(master)
        per_group_temps[(t, e)].extend(temps)

    dark_maps: dict[tuple[float, float], np.ndarray] = {}
    for key, masters in tqdm(per_group_masters.items(), desc="Writing dark master per group"):
        stack = np.stack(masters, axis=0)
        temps = per_group_temps[key]
        master = np.mean(stack, axis=0)
        hdr = fits.Header()
        hdr["NSOURCE"] = stack.shape[0]
        hdr["MEAN"] = float(np.mean(stack))
        hdr["MEDIAN"] = float(np.median(stack))
        hdr["STD"] = float(np.std(stack))
        hdr["DATAMIN"] = float(np.min(stack))
        hdr["DATAMAX"] = float(np.max(stack))
        if temps:
            hdr["TMIN"] = float(np.min(temps))
            hdr["TMAX"] = float(np.max(temps))
            hdr["TAVG"] = float(np.mean(temps))
        t, e = key
        out_name = f"master_dark_T{t:.1f}_E{e:.1f}.fits"
        fits.writeto(os.path.join(outdir_dark, out_name), master.astype(np.float32), hdr, overwrite=True)
        dark_maps[key] = master

    flat_groups: dict[tuple[float, float, str], list[dict]] = defaultdict(list)
    for _, row in tqdm(flat_df.iterrows(), total=len(flat_df), desc="Grouping flat frames"):
        eq_temp, p_exp = _parse_temp_exp_from_path(row["PATH"])
        hdr = fits.getheader(row["PATH"])
        exp = p_exp if p_exp is not None else hdr.get("EXPTIME")
        temp = eq_temp if eq_temp is not None else row["TEMP"]
        attempt = _attempt_from_path(row["PATH"])
        flat_groups[(temp, exp, attempt)].append({"path": row["PATH"], "temp": row["TEMP"], "exp": exp})

    per_flat_masters: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    per_flat_temps: dict[tuple[float, float], list[float]] = defaultdict(list)

    for (t, e, attempt), entries in tqdm(flat_groups.items(), desc="Combining flat per attempt"):
        paths = [e["path"] for e in entries]
        temps = [e["temp"] for e in entries]
        exps = [e["exp"] for e in entries]
        master, hdr = _make_mean_master(paths, temps=temps, exps=exps)
        name = f"master_flat_{attempt}_T{t:.1f}_E{e:.1f}.fits"
        fits.writeto(os.path.join(outdir_flat, name), master.astype(np.float32), hdr, overwrite=True)
        per_flat_masters[(t, e)].append(master)
        per_flat_temps[(t, e)].extend(temps)

    flat_maps: dict[tuple[float, float], np.ndarray] = {}
    for key, masters in tqdm(per_flat_masters.items(), desc="Writing flat master per group"):
        stack = np.stack(masters, axis=0)
        temps = per_flat_temps[key]
        master = np.mean(stack, axis=0)
        hdr = fits.Header()
        hdr["NSOURCE"] = stack.shape[0]
        hdr["MEAN"] = float(np.mean(stack))
        hdr["MEDIAN"] = float(np.median(stack))
        hdr["STD"] = float(np.std(stack))
        hdr["DATAMIN"] = float(np.min(stack))
        hdr["DATAMAX"] = float(np.max(stack))
        if temps:
            hdr["TMIN"] = float(np.min(temps))
            hdr["TMAX"] = float(np.max(temps))
            hdr["TAVG"] = float(np.mean(temps))
        t, e = key
        out_name = f"master_flat_T{t:.1f}_E{e:.1f}.fits"
        fits.writeto(os.path.join(outdir_flat, out_name), master.astype(np.float32), hdr, overwrite=True)
        flat_maps[key] = master

    return dark_maps, flat_maps


def compute_stats(df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """Compute basic statistics for each FITS frame and save to CSV."""
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing stats"):
        with fits.open(row["PATH"]) as hdul:
            data = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
        stats = {
            "PATH": row["PATH"],
            "CALTYPE": row["CALTYPE"],
            "TEMP": row["TEMP"],
            "EXPTIME": hdr.get("EXPTIME"),
            "MEAN": float(np.mean(data)),
            "MEDIAN": float(np.median(data)),
            "STD": float(np.std(data)),
            "P16": float(np.percentile(data, 16)),
            "P84": float(np.percentile(data, 84)),
        }
        records.append(stats)
    df_stats = pd.DataFrame.from_records(records)
    df_stats.to_csv(out_csv, index=False)
    return df_stats


def plot_bias_trend(stats: pd.DataFrame, outdir: str) -> None:
    """Generate mean/median/std vs temperature plots for bias frames."""
    os.makedirs(outdir, exist_ok=True)
    bias = stats[stats["CALTYPE"] == "BIAS"]
    if bias.empty:
        return

    for metric in ["MEAN", "MEDIAN", "STD"]:
        plt.figure()
        plt.scatter(bias["TEMP"], bias[metric])
        plt.xlabel("Temperature (°C)")
        plt.ylabel(f"{metric} ADU")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bias_{metric.lower()}_vs_temp.png"))
        plt.close()


def plot_dark_flat_trends(stats: pd.DataFrame, outdir: str, caltype: str) -> None:
    """Plot mean/median/std vs exposure time for dark or flat frames."""
    os.makedirs(outdir, exist_ok=True)
    df = stats[stats["CALTYPE"] == caltype]
    if df.empty:
        return

    for temp in sorted(df["TEMP"].dropna().unique()):
        sub = df[df["TEMP"] == temp]
        for metric in ["MEAN", "MEDIAN", "STD"]:
            plt.figure()
            plt.scatter(sub["EXPTIME"], sub[metric])
            plt.xlabel("Exposure Time [s]")
            plt.ylabel(f"{metric} ADU")
            plt.title(f"T={temp:.1f}°C")
            plt.tight_layout()
            fname = f"{caltype.lower()}_{metric.lower()}_T{temp:.1f}.png"
            plt.savefig(os.path.join(outdir, fname))
            plt.close()


def _find_outlier(stats: pd.DataFrame, caltype: str, temp: float, exptime: float | None = None) -> str | None:
    """Return the path of the frame with the highest STD for the given group."""
    group = stats[(stats["CALTYPE"] == caltype) & (stats["TEMP"] == temp)]
    if exptime is not None:
        group = group[np.isclose(group["EXPTIME"], exptime)]
    if group.empty:
        return None
    idx = group["STD"].idxmax()
    return group.loc[idx, "PATH"]


def save_comparison_images(
    master_dict: dict,
    stats: pd.DataFrame,
    outdir: str,
    caltype: str,
) -> None:
    """Save comparison between each master frame and the worst outlier frame."""
    os.makedirs(outdir, exist_ok=True)
    for key, master in tqdm(
        master_dict.items(), desc=f"Saving {caltype} comparisons"
    ):
        if isinstance(key, tuple):
            temp, exptime = key
        else:
            temp, exptime = key, None
        outlier_path = _find_outlier(stats, caltype, temp, exptime)
        if outlier_path is None:
            continue
        outlier = _load_fits(outlier_path)
        vmin, vmax = np.percentile(master, [5, 95])
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(master, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.title("Master")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(outlier, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.title("Outlier")
        plt.axis("off")
        plt.tight_layout()
        if exptime is None:
            fname = f"{caltype.lower()}_T{temp:.1f}.png"
        else:
            fname = f"{caltype.lower()}_T{temp:.1f}_E{exptime:.1f}.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main(index_csv: str, output_dir: str) -> None:
    bias_df, dark_df, flat_df = load_index(index_csv)

    master_dir = os.path.join(output_dir, "masters")
    bias_masters = master_bias_by_temp(bias_df, os.path.join(master_dir, "bias"))
    dark_masters, flat_masters = master_dark_flat(
        dark_df, flat_df,
        os.path.join(master_dir, "darks"),
        os.path.join(master_dir, "flats"),
    )

    stats_csv = os.path.join(output_dir, "frame_stats.csv")
    stats = compute_stats(pd.concat([bias_df, dark_df, flat_df], ignore_index=True), stats_csv)

    plots_dir = os.path.join(output_dir, "plots")
    plot_bias_trend(stats, plots_dir)
    plot_dark_flat_trends(stats, plots_dir, "DARK")
    plot_dark_flat_trends(stats, plots_dir, "FLAT")

    comp_dir = os.path.join(output_dir, "comparisons")
    save_comparison_images(bias_masters, stats, comp_dir, "BIAS")
    save_comparison_images(dark_masters, stats, comp_dir, "DARK")
    save_comparison_images(flat_masters, stats, comp_dir, "FLAT")

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process calibration index CSV")
    parser.add_argument("index_csv", help="Path to index.csv generated by utils.index_dataset")
    parser.add_argument("output_dir", help="Directory to store results")
    args = parser.parse_args()
    main(args.index_csv, args.output_dir)
