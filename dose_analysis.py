#!/usr/bin/env python3
"""Analyse calibration frames grouped by radiation dose.

This script reads an ``index.csv`` file produced by
``utils.index_dataset`` and generates master calibration
frames grouped by radiation stage, type, exposure time and
radiation dose extracted from the file paths.  Each master
FITS includes temperature statistics and per-frame metrics in
its header.  A ``dose_summary.csv`` table summarises the mean
and standard deviation of every master.  Finally, trend plots
of the mean signal versus radiation dose are generated for
BIAS, DARK and FLAT frames.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

from operation_analysis import _parse_rads
from process_index import _make_mean_master, _parse_temp_exp_from_path


def _dose_from_path(path: str) -> float | None:
    """Return the radiation dose encoded in *path* or ``None``."""
    for part in path.split(os.sep):
        val = _parse_rads(part)
        if val is not None:
            return val
    return None


def _exptime_from_path(path: str) -> float | None:
    """Return the exposure time for *path*.

    The value is first parsed from the filename and, if missing,
    read from the FITS header.
    """
    _, exp = _parse_temp_exp_from_path(path)
    if exp is not None:
        return exp

    base = os.path.basename(path)
    m = re.search(r"_E([0-9]+(?:\.[0-9]+)?)", base)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    try:
        hdr = fits.getheader(path)
        if "EXPTIME" in hdr:
            return float(hdr["EXPTIME"])
    except Exception:
        pass
    return None


def _temperature_from_header(path: str) -> float | None:
    try:
        hdr = fits.getheader(path)
        if "TEMP" in hdr:
            return float(hdr["TEMP"])
    except Exception:
        pass
    return None


def _make_master(paths: List[str]) -> Tuple[np.ndarray, fits.Header]:
    """Compute mean master frame and header with extended statistics."""
    stack = []
    temps = []
    means = []
    stds = []
    for p in paths:
        data = fits.getdata(p).astype(np.float32)
        stack.append(data)
        temps.append(_temperature_from_header(p))
        means.append(float(np.mean(data)))
        stds.append(float(np.std(data)))
    stack_arr = np.stack(stack, axis=0)
    master, hdr = _make_mean_master(paths)
    valid_temps = [t for t in temps if t is not None and np.isfinite(t)]
    if valid_temps:
        hdr["T_MEAN"] = float(np.mean(valid_temps))
        hdr["T_STD"] = float(np.std(valid_temps))
    for idx, (src, m, s) in enumerate(zip(paths, means, stds), start=1):
        hdr[f"SRC{idx:03d}"] = os.path.basename(src)
        hdr[f"M{idx:03d}"] = m
        hdr[f"S{idx:03d}"] = s
    return master, hdr


def _group_paths(df: pd.DataFrame) -> Dict[Tuple[str, str, float, float | None], List[str]]:
    groups: Dict[Tuple[str, str, float, float | None], List[str]] = defaultdict(list)
    during_doses: List[float] = []

    for _, row in df.iterrows():
        if row["STAGE"] == "during":
            d = _dose_from_path(row["PATH"])
            if d is not None:
                during_doses.append(d)

    min_dose = min(during_doses) if during_doses else 0.0
    max_dose = max(during_doses) if during_doses else 0.0

    for _, row in df.iterrows():
        stage = row["STAGE"]
        cal = row["CALTYPE"]
        dose = _dose_from_path(row["PATH"])
        if stage == "pre":
            dose = min_dose
        elif stage == "post":
            dose = max_dose
        if dose is None:
            dose = 0.0
        exp = _exptime_from_path(row["PATH"])
        groups[(stage, cal, dose, exp)].append(row["PATH"])
    return groups


def _save_plot(summary: pd.DataFrame, outdir: str) -> None:
    """Plot mean signal vs dose for all stages grouped by exposure time."""
    if summary.empty:
        return

    os.makedirs(outdir, exist_ok=True)
    for cal in sorted(summary["CALTYPE"].unique()):
        cal_df = summary[summary["CALTYPE"] == cal]
        if cal_df.empty:
            continue

        exp_values = sorted(cal_df["EXPTIME"].dropna().unique())
        if not exp_values:
            exp_values = [None]

        for exp in exp_values:
            if exp is None:
                sub = cal_df[cal_df["EXPTIME"].isna()]
            else:
                sub = cal_df[cal_df["EXPTIME"] == exp]

            if sub.empty:
                continue

            plt.figure()
            for stage in ("pre", "during", "post"):
                stage_df = sub[sub["STAGE"] == stage]
                if stage_df.empty:
                    continue
                x = stage_df["DOSE"].astype(float)
                y = stage_df["MEAN"].astype(float)
                e = stage_df["STD"].astype(float)
                order = np.argsort(x)
                x = x.iloc[order]
                y = y.iloc[order]
                e = e.iloc[order]
                fmt = "o" if len(x) == 1 else "-o"
                plt.errorbar(x, y, yerr=e, fmt=fmt, label=stage)

            plt.xlabel("Dose [kRad]")
            plt.ylabel("Mean ADU")
            title = cal
            if exp is not None:
                title += f" E={exp:g}s"
            plt.title(title)
            plt.legend()
            plt.tight_layout()

            fname = f"{cal.lower()}_mean_vs_dose"
            if exp is not None:
                fname += f"_E{str(exp).replace('.', 'p')}s"
            fname += ".png"
            plt.savefig(os.path.join(outdir, fname))
            plt.close()


def _compute_photometric_precision(summary: pd.DataFrame) -> pd.DataFrame:
    """Return estimated photometric precision for ``during`` bias/dark."""
    df = summary[summary["STAGE"] == "during"]
    bias = df[df["CALTYPE"] == "BIAS"]
    dark = df[df["CALTYPE"] == "DARK"]
    doses = sorted(set(bias["DOSE"]) & set(dark["DOSE"]))
    full_scale = 4096 * 16.0  # ADU
    rows = []
    for d in doses:
        b = bias[bias["DOSE"] == d]
        dk = dark[dark["DOSE"] == d]
        b_mean = b["MEAN"].mean()
        b_std = b["STD"].mean()
        d_std = dk["STD"].mean()
        signal = (full_scale - b_mean) / 2.0
        noise = np.sqrt(signal + b_std**2 + d_std**2)
        snr = signal / noise if noise > 0 else 0.0
        mag_err = 1.0857 / snr if snr > 0 else np.inf
        rows.append({"DOSE": float(d), "MAG_ERR": float(mag_err)})
    return pd.DataFrame(rows)


def _plot_photometric_precision(df: pd.DataFrame, outdir: str) -> None:
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    df = df.sort_values("DOSE")
    plt.figure()
    plt.plot(df["DOSE"], df["MAG_ERR"], marker="o")
    plt.xlabel("Dose [kRad]")
    plt.ylabel("Magnitude error [mag]")
    plt.title("Photometric precision during irradiation")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "photometric_precision_vs_dose.png"))
    plt.close()

def main(index_csv: str, output_dir: str) -> None:
    df = pd.read_csv(index_csv)
    df = df[df.get("BADFITS", False) == False]

    groups = _group_paths(df)

    master_dir = os.path.join(output_dir, "masters")
    os.makedirs(master_dir, exist_ok=True)

    records = []
    for (stage, cal, dose, exp), paths in groups.items():
        master, hdr = _make_master(paths)
        name = f"master_{cal.lower()}_{stage}_D{dose:g}kR_E{exp if exp is not None else 'none'}".replace('/', '_')
        fpath = os.path.join(master_dir, name + ".fits")
        fits.writeto(fpath, master.astype(np.float32), hdr, overwrite=True)
        records.append({
            "STAGE": stage,
            "CALTYPE": cal,
            "DOSE": dose,
            "EXPTIME": exp,
            "MEAN": hdr["MEAN"],
            "STD": hdr["STD"],
        })

    summary = pd.DataFrame.from_records(records)
    summary_csv = os.path.join(output_dir, "dose_summary.csv")
    summary.to_csv(summary_csv, index=False)

    _save_plot(summary, os.path.join(output_dir, "plots"))

    precision_df = _compute_photometric_precision(summary)
    precision_csv = os.path.join(output_dir, "photometric_precision.csv")
    precision_df.to_csv(precision_csv, index=False)
    _plot_photometric_precision(precision_df, os.path.join(output_dir, "plots"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse calibration frames by radiation dose")
    parser.add_argument("index_csv", help="Path to index.csv")
    parser.add_argument("output_dir", help="Directory for results")
    args = parser.parse_args()
    main(args.index_csv, args.output_dir)
