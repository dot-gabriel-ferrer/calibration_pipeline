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
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

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
    for p in tqdm(paths, desc="Reading frames", unit="frame", leave=False):
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


def _stack_stats(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-pixel mean and standard deviation for *paths*."""
    stack = []
    for p in tqdm(paths, desc="Stacking", unit="frame", leave=False):
        stack.append(fits.getdata(p).astype(np.float32))
    arr = np.stack(stack, axis=0)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return mean, std


def _group_paths(df: pd.DataFrame) -> Dict[Tuple[str, str, float, float | None], List[str]]:
    groups: Dict[Tuple[str, str, float, float | None], List[str]] = defaultdict(list)
    during_doses: List[float] = []
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Scanning doses",
        unit="file",
        leave=False,
    ):
        if row["STAGE"] == "radiating":
            d = _dose_from_path(row["PATH"])
            if d is not None:
                during_doses.append(d)

    min_dose = min(during_doses) if during_doses else 0.0
    max_dose = max(during_doses) if during_doses else 0.0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Grouping paths",
        unit="file",
        leave=False,
    ):
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

            fig, ax = plt.subplots()
            stats_lines = []
            plot_data = {}
            for stage in ("pre", "radiating", "post"):
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
                ax.errorbar(x, y, yerr=e, fmt=fmt, label=stage)
                ax.fill_between(x, y - e, y + e, alpha=0.2)
                stats_lines.append(f"{stage}: \u03BC={y.mean():.1f}, \u03C3={e.mean():.1f}")
                plot_data[f"{stage}_dose"] = x.to_numpy()
                plot_data[f"{stage}_mean"] = y.to_numpy()
                plot_data[f"{stage}_std"] = e.to_numpy()

            ax.set_xlabel("Dose [kRad]")
            ax.set_ylabel("Mean ADU")
            title = cal
            if exp is not None:
                title += f" E={exp:g}s"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.text(1.02, 0.5, "\n".join(stats_lines), va="center")
            fig.tight_layout()

            fname = f"{cal.lower()}_mean_vs_dose"
            if exp is not None:
                fname += f"_E{str(exp).replace('.', 'p')}s"
            fname += ".png"
            out_png = os.path.join(outdir, fname)
            fig.savefig(out_png)
            plt.close(fig)

            # Save the arrays used for the plot
            np.savez_compressed(os.path.splitext(out_png)[0] + ".npz", **plot_data)


def _compute_photometric_precision(summary: pd.DataFrame) -> pd.DataFrame:
    """Return estimated photometric precision for ``radiating`` bias/dark.

    The returned ``DataFrame`` now also includes the standard deviation of
    the magnitude error for each dose (``MAG_ERR_STD``).  The deviation is
    computed from the magnitude error estimated for every available bias/dark
    frame combination at the same radiation dose.
    """

    df = summary[summary["STAGE"] == "radiating"]
    bias = df[df["CALTYPE"] == "BIAS"]
    dark = df[df["CALTYPE"] == "DARK"]
    doses = sorted(set(bias["DOSE"]) & set(dark["DOSE"]))
    full_scale = 4096 * 16.0  # ADU
    rows = []
    for d in tqdm(doses, desc="Per-dose analysis", unit="dose"):
        b = bias[bias["DOSE"] == d]
        dk = dark[dark["DOSE"] == d]

        mag_errs: list[float] = []
        for _, b_row in b.iterrows():
            for _, d_row in dk.iterrows():
                signal = (full_scale - b_row["MEAN"]) / 2.0
                noise = np.sqrt(signal + b_row["STD"] ** 2 + d_row["STD"] ** 2)
                snr = signal / noise if noise > 0 else 0.0
                mag_err = 1.0857 / snr if snr > 0 else np.inf
                mag_errs.append(float(mag_err))

        if mag_errs:
            rows.append(
                {
                    "DOSE": float(d),
                    "MAG_ERR": float(np.mean(mag_errs)),
                    "MAG_ERR_STD": float(np.std(mag_errs)),
                }
            )

    return pd.DataFrame(rows)


def _plot_photometric_precision(df: pd.DataFrame, outdir: str) -> None:
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    df = df.sort_values("DOSE")
    fig, ax = plt.subplots()
    ax.errorbar(
        df["DOSE"],
        df["MAG_ERR"],
        yerr=df.get("MAG_ERR_STD"),
        fmt="o-",
    )
    ax.set_xlabel("Dose [kRad]")
    ax.set_ylabel("Magnitude error [mag]")
    ax.set_title("Photometric precision during irradiation")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    stats = (
        f"min={df['MAG_ERR'].min():.3f}\n"
        f"max={df['MAG_ERR'].max():.3f}\n"
        f"\u03C3={df.get('MAG_ERR_STD').mean():.3f}"
    )
    fig.text(1.02, 0.5, stats, va="center")
    fig.tight_layout()
    out_png = os.path.join(outdir, "photometric_precision_vs_dose.png")
    fig.savefig(out_png)
    plt.close(fig)

    # Save arrays for further analysis
    data = {
        "dose": df["DOSE"].to_numpy(float),
        "mag_err": df["MAG_ERR"].to_numpy(float),
    }
    if "MAG_ERR_STD" in df:
        data["mag_err_std"] = df["MAG_ERR_STD"].to_numpy(float)
    np.savez_compressed(os.path.splitext(out_png)[0] + ".npz", **data)


def _plot_error_vs_dose(df: pd.DataFrame, outdir: str) -> None:
    """Plot global magnitude/ADU errors as a function of dose."""
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    df = df.sort_values("DOSE")

    fig_m, ax_m = plt.subplots()
    ax_m.errorbar(
        df["DOSE"],
        df["MAG_MEAN"],
        yerr=df.get("MAG_STD"),
        fmt="o-",
    )
    ax_m.fill_between(
        df["DOSE"],
        df["MAG_MEAN"] - df.get("MAG_STD", 0),
        df["MAG_MEAN"] + df.get("MAG_STD", 0),
        alpha=0.2,
    )
    ax_m.set_xlabel("Dose [kRad]")
    ax_m.set_ylabel("Magnitude error [mag]")
    ax_m.set_title("Magnitude error vs dose")
    ax_m.grid(True, ls="--", alpha=0.5)
    fig_m.tight_layout()
    out_m = os.path.join(outdir, "mag_err_vs_dose.png")
    fig_m.savefig(out_m)
    plt.close(fig_m)

    np.savez_compressed(
        os.path.splitext(out_m)[0] + ".npz",
        dose=df["DOSE"].to_numpy(float),
        mag_mean=df["MAG_MEAN"].to_numpy(float),
        mag_std=df.get("MAG_STD", pd.Series()).to_numpy(float) if "MAG_STD" in df else np.empty(0),
    )

    fig_a, ax_a = plt.subplots()
    ax_a.errorbar(
        df["DOSE"],
        df["ADU_MEAN"],
        yerr=df.get("ADU_STD"),
        fmt="o-",
    )
    ax_a.fill_between(
        df["DOSE"],
        df["ADU_MEAN"] - df.get("ADU_STD", 0),
        df["ADU_MEAN"] + df.get("ADU_STD", 0),
        alpha=0.2,
    )
    ax_a.set_xlabel("Dose [kRad]")
    ax_a.set_ylabel("ADU error (16 bit)")
    ax_a.set_title("ADU error vs dose")
    ax_a.grid(True, ls="--", alpha=0.5)
    fig_a.tight_layout()
    out_a = os.path.join(outdir, "adu_err_vs_dose.png")
    fig_a.savefig(out_a)
    plt.close(fig_a)

    np.savez_compressed(
        os.path.splitext(out_a)[0] + ".npz",
        dose=df["DOSE"].to_numpy(float),
        adu_mean=df["ADU_MEAN"].to_numpy(float),
        adu_std=df.get("ADU_STD", pd.Series()).to_numpy(float) if "ADU_STD" in df else np.empty(0),
    )


def _pixel_precision_analysis(
    groups: Dict[Tuple[str, str, float, float | None], List[str]], outdir: str
) -> pd.DataFrame:
    """Generate per-pixel magnitude and ADU error maps for each dose."""
    bias_groups = {k: v for k, v in groups.items() if k[0] == "radiating" and k[1] == "BIAS"}
    dark_groups = {k: v for k, v in groups.items() if k[0] == "radiating" and k[1] == "DARK"}
    doses = sorted(set(k[2] for k in bias_groups) & set(k[2] for k in dark_groups))
    if not doses:
        return

    os.makedirs(outdir, exist_ok=True)
    zone_labels = ["Q1", "Q2", "Q3", "Q4"]
    zone_mag: Dict[str, List[Tuple[float, float]]] = {z: [] for z in zone_labels}
    zone_adu: Dict[str, List[Tuple[float, float]]] = {z: [] for z in zone_labels}
    stats_rows: List[dict[str, float]] = []

    for d in tqdm(doses, desc="Per-dose pixel analysis", unit="dose"):
        b_paths = [p for k, v in bias_groups.items() if k[2] == d for p in v]
        d_paths = [p for k, v in dark_groups.items() if k[2] == d for p in v]
        if not b_paths or not d_paths:
            continue

        b_mean, b_std = _stack_stats(b_paths)
        _, d_std = _stack_stats(d_paths)
        full_scale = 4096 * 16.0
        signal = (full_scale - b_mean) / 2.0
        noise = np.sqrt(np.maximum(signal, 0) + b_std ** 2 + d_std ** 2)
        snr = np.where(noise > 0, signal / noise, 0.0)
        mag_err = np.where(snr > 0, 1.0857 / snr, np.inf)
        adu_err16 = noise
        adu_err12 = noise / 16.0

        mag_err_mean = float(np.mean(mag_err))
        adu_err16_mean = float(np.mean(adu_err16))
        adu_err12_mean = float(np.mean(adu_err12))

        mag_norm = (mag_err - mag_err_mean) / mag_err_mean
        adu16_norm = (adu_err16 - adu_err16_mean) / adu_err16_mean
        adu12_norm = (adu_err12 - adu_err12_mean) / adu_err12_mean

        tag = f"{d:g}kR"
        fits.writeto(os.path.join(outdir, f"mag_err_{tag}.fits"), mag_err.astype(np.float32), overwrite=True)
        fits.writeto(os.path.join(outdir, f"adu_err16_{tag}.fits"), adu_err16.astype(np.float32), overwrite=True)
        fits.writeto(os.path.join(outdir, f"adu_err12_{tag}.fits"), adu_err12.astype(np.float32), overwrite=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mag_norm, origin="lower", cmap="magma")
        cbar = plt.colorbar(im, ax=ax, label="Magnitude error [mag]"+f" mean={mag_err_mean:.2f}")
        #cbar.ax.text(1.05, 0.5, f"mean={mag_err_mean:.2f}", transform=cbar.ax.transAxes, va="center")
        ax.set_title(f"Magnitude error {tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"mag_err_{tag}.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(adu16_norm, origin="lower", cmap="viridis")
        cbar = plt.colorbar(im, ax=ax, label="ADU error (16 bit)"+f" mean={adu_err16_mean:.2f}")
        #cbar.ax.text(1.05, 0.5, f"mean={adu_err16_mean:.2f}", transform=cbar.ax.transAxes, va="center")
        ax.set_title(f"ADU error 16-bit {tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"adu_err16_{tag}.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(adu12_norm, origin="lower", cmap="viridis")
        cbar = plt.colorbar(im, ax=ax, label="ADU error (12 bit)"+f" mean={adu_err12_mean:.2f}")
        cbar.ax.text(1.05, 0.5, f"mean={adu_err12_mean:.2f}", transform=cbar.ax.transAxes, va="center")
        ax.set_title(f"ADU error 12-bit {tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"adu_err12_{tag}.png"), dpi=300)
        plt.close(fig)

        h, w = mag_err.shape
        my, mx = h // 2, w // 2
        zones = [
            (slice(0, my), slice(0, mx)),
            (slice(0, my), slice(mx, None)),
            (slice(my, None), slice(0, mx)),
            (slice(my, None), slice(mx, None)),
        ]
        for label, (ys, xs) in zip(zone_labels, zones):
            zone_mag[label].append((d, float(np.mean(mag_err[ys, xs]))))
            zone_adu[label].append((d, float(np.mean(adu_err16[ys, xs]))))

        stats_rows.append(
            {
                "DOSE": float(d),
                "MAG_MEAN": float(np.mean(mag_err)),
                "MAG_STD": float(np.std(mag_err)),
                "ADU_MEAN": float(np.mean(adu_err16)),
                "ADU_STD": float(np.std(adu_err16)),
            }
        )

    # Zone plots
    fig_m, ax_m = plt.subplots()
    for label, vals in zone_mag.items():
        if not vals:
            continue
        doses = [v[0] for v in vals]
        errs = [v[1] for v in vals]
        ax_m.plot(doses, errs, marker="o", label=label)
    ax_m.set_xlabel("Dose [kRad]")
    ax_m.set_ylabel("Magnitude error [mag]")
    ax_m.set_title("Per-zone magnitude error")
    ax_m.legend()
    ax_m.grid(True, ls="--", alpha=0.5)
    fig_m.tight_layout()
    fig_m.savefig(os.path.join(outdir, "zone_mag_error.png"))
    plt.close(fig_m)

    fig_a, ax_a = plt.subplots()
    for label, vals in zone_adu.items():
        if not vals:
            continue
        doses = [v[0] for v in vals]
        errs = [v[1] for v in vals]
        ax_a.plot(doses, errs, marker="o", label=label)
    ax_a.set_xlabel("Dose [kRad]")
    ax_a.set_ylabel("ADU error (16 bit)")
    ax_a.set_title("Per-zone ADU error")
    ax_a.legend()
    ax_a.grid(True, ls="--", alpha=0.5)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(outdir, "zone_adu_error.png"))
    plt.close(fig_a)

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(outdir, "pixel_precision_stats.csv"), index=False)
        _plot_error_vs_dose(stats_df, outdir)
    return stats_df


def _master_path(cal: str, stage: str, dose: float, exp: float | None, master_dir: str) -> str:
    """Return the path to a master frame with the given parameters."""
    name = f"master_{cal.lower()}_{stage}_D{dose:g}kR_E{exp if exp is not None else 'none'}"
    return os.path.join(master_dir, name.replace('/', '_') + ".fits")


def _compare_stage_differences(summary: pd.DataFrame, master_dir: str, outdir: str) -> None:
    """Store differences between initial/last irradiation values and pre/post.

    Heatmaps now display the percentage change relative to the reference mean
    value with a symmetric colour scale.
    """
    df = summary
    during = df[df["STAGE"] == "radiating"]
    pre = df[df["STAGE"] == "pre"]
    post = df[df["STAGE"] == "post"]
    if during.empty:
        return
    min_dose = during["DOSE"].min()
    max_dose = during["DOSE"].max()

    rows = []
    os.makedirs(outdir, exist_ok=True)
    for cal in ("BIAS", "DARK"):
        dmin = during[(during["CALTYPE"] == cal) & (during["DOSE"] == min_dose)]
        dmax = during[(during["CALTYPE"] == cal) & (during["DOSE"] == max_dose)]
        p_pre = pre[pre["CALTYPE"] == cal]
        p_post = post[post["CALTYPE"] == cal]
        if not dmin.empty and not p_pre.empty:
            diff = float(dmin["MEAN"].mean() - p_pre["MEAN"].mean())
            rows.append({"CALTYPE": cal, "CMP": "first_vs_pre", "DIFF": diff})

            # Heatmap of first during minus pre
            ref_row = p_pre.iloc[0]
            targ_row = dmin.iloc[0]
            ref_path = _master_path(cal, "pre", ref_row["DOSE"], ref_row["EXPTIME"], master_dir)
            targ_path = _master_path(cal, "radiating", targ_row["DOSE"], targ_row["EXPTIME"], master_dir)
            if os.path.isfile(ref_path) and os.path.isfile(targ_path):
                ref = fits.getdata(ref_path)
                targ = fits.getdata(targ_path)
                diff_img = targ - ref
                base_val = float(ref_row["MEAN"])
                if base_val:
                    diff_img = (diff_img / base_val) * 100.0
                    label = f"% change (base={base_val:.2f} ADU)"
                else:
                    label = "ADU"
                vmin = float(np.nanmin(diff_img))
                vmax = float(np.nanmax(diff_img))
                plt.figure(figsize=(6, 5))
                im = plt.imshow(
                    diff_img,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(im, label=label)
                plt.title(f"{cal} first vs pre")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{cal.lower()}_first_vs_pre.png"))
                plt.close()

                abs_max = max(abs(vmin), abs(vmax))
                log_norm = SymLogNorm(linthresh=abs_max * 0.01 + 1e-9, vmin=-abs_max, vmax=abs_max)
                plt.figure(figsize=(6, 5))
                im = plt.imshow(
                    diff_img,
                    origin="lower",
                    cmap="coolwarm",
                    norm=log_norm,
                )
                plt.colorbar(im, label=label)
                plt.title(f"{cal} first vs pre (log)")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{cal.lower()}_first_vs_pre_log.png"))
                plt.close()
        if not dmax.empty and not p_post.empty:
            diff = float(p_post["MEAN"].mean() - dmax["MEAN"].mean())
            rows.append({"CALTYPE": cal, "CMP": "post_vs_last", "DIFF": diff})

            # Heatmap of post minus last during
            ref_row = dmax.iloc[0]
            targ_row = p_post.iloc[0]
            ref_path = _master_path(cal, "radiating", ref_row["DOSE"], ref_row["EXPTIME"], master_dir)
            targ_path = _master_path(cal, "post", targ_row["DOSE"], targ_row["EXPTIME"], master_dir)
            if os.path.isfile(ref_path) and os.path.isfile(targ_path):
                ref = fits.getdata(ref_path)
                targ = fits.getdata(targ_path)
                diff_img = targ - ref
                base_val = float(ref_row["MEAN"])
                if base_val:
                    diff_img = (diff_img / base_val) * 100.0
                    label = f"% change (base={base_val:.2f} ADU)"
                else:
                    label = "ADU"
                vmin = float(np.nanmin(diff_img))
                vmax = float(np.nanmax(diff_img))
                plt.figure(figsize=(6, 5))
                im = plt.imshow(
                    diff_img,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(im, label=label)
                plt.title(f"{cal} post vs last")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{cal.lower()}_post_vs_last.png"))
                plt.close()

                abs_max = max(abs(vmin), abs(vmax))
                log_norm = SymLogNorm(linthresh=abs_max * 0.01 + 1e-9, vmin=-abs_max, vmax=abs_max)
                plt.figure(figsize=(6, 5))
                im = plt.imshow(
                    diff_img,
                    origin="lower",
                    cmap="coolwarm",
                    norm=log_norm,
                )
                plt.colorbar(im, label=label)
                plt.title(f"{cal} post vs last (log)")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{cal.lower()}_post_vs_last_log.png"))
                plt.close()

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "stage_differences.csv"), index=False)


def _fit_dose_response(summary: pd.DataFrame, outdir: str) -> None:
    """Fit a linear model of mean value vs dose and store coefficients."""
    df = summary[summary["STAGE"] == "radiating"]
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for cal in df["CALTYPE"].unique():
        cal_df = df[df["CALTYPE"] == cal]

        if cal == "DARK":
            exp_values = list(sorted(cal_df["EXPTIME"].dropna().unique()))
            groups = [(exp, cal_df[np.isclose(cal_df["EXPTIME"], exp)]) for exp in exp_values]
            if cal_df["EXPTIME"].isna().any():
                groups.append((None, cal_df[cal_df["EXPTIME"].isna()]))
        else:
            groups = [(None, cal_df)]

        for exp, sub in groups:
            if len(sub) < 2:
                continue
            x = sub["DOSE"].astype(float)
            y = sub["MEAN"].astype(float)
            coeff = np.polyfit(x, y, 1)

            row = {
                "CALTYPE": cal,
                "SLOPE": float(coeff[0]),
                "INTERCEPT": float(coeff[1]),
            }
            if exp is not None:
                row["EXPTIME"] = float(exp)
            rows.append(row)

            # Plot data, fitted line and residuals
            fig, (ax_top, ax_resid) = plt.subplots(2, 1, sharex=True)
            ax_top.scatter(x, y, label="data")
            xfit = np.linspace(float(x.min()), float(x.max()), 100)
            yfit = coeff[0] * xfit + coeff[1]
            ax_top.plot(xfit, yfit, color="C1", label="fit")
            ax_top.set_ylabel("Mean ADU")
            ax_top.legend()
    
            resid = y - (coeff[0] * x + coeff[1])
            ax_resid.scatter(x, resid)
            ax_resid.axhline(0.0, color="C1", ls="--")
            ax_resid.set_xlabel("Dose [kRad]")
            ax_resid.set_ylabel("Residual ADU")
    
            fig.suptitle(f"{cal} mean vs dose" + (f" E={exp:g}s" if exp is not None else ""))
            fig.tight_layout()
            fname = f"dose_model_{cal.lower()}"
            if exp is not None:
                fname += f"_E{str(exp).replace('.', 'p')}s"
            fig.savefig(os.path.join(outdir, fname + ".png"))
            plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "dose_model.csv"), index=False)


def _fit_base_level_trend(summary: pd.DataFrame, outdir: str) -> None:
    """Fit a linear model of mean bias/dark level vs dose.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output table from :func:`main` with at least the ``STAGE``, ``CALTYPE``,
        ``DOSE`` and ``MEAN`` columns.
    outdir : str
        Directory where the CSV and plots will be stored.
    """

    df = summary[summary["STAGE"] == "radiating"]
    if df.empty:
        return

    os.makedirs(outdir, exist_ok=True)
    rows: list[dict[str, float]] = []

    for cal in ("BIAS", "DARK"):
        cal_df = df[df["CALTYPE"] == cal]
        if cal_df.empty:
            continue
        grouped = cal_df.groupby("DOSE")["MEAN"].mean().reset_index()
        if len(grouped) < 2:
            continue

        x = grouped["DOSE"].astype(float)
        y = grouped["MEAN"].astype(float)
        coeff = np.polyfit(x, y, 1)

        rows.append(
            {
                "CALTYPE": cal,
                "SLOPE": float(coeff[0]),
                "INTERCEPT": float(coeff[1]),
            }
        )

        fig, ax = plt.subplots()
        ax.scatter(x, y, label="mean")
        xfit = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xfit, coeff[0] * xfit + coeff[1], color="C1", label="fit")
        ax.set_xlabel("Dose [kRad]")
        ax.set_ylabel("Mean ADU")
        ax.set_title(f"{cal} mean vs dose")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"base_level_trend_{cal.lower()}.png"))
        plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(outdir, "base_level_trend.csv"), index=False
        )


def _stage_base_level_diff(summary: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Compare pre/post base levels with first/last irradiation values.

    Parameters
    ----------
    summary : pandas.DataFrame
        Table from :func:`main` with at least ``STAGE``, ``CALTYPE``, ``DOSE``
        and ``MEAN`` columns.
    outdir : str
        Directory for the plot and saved arrays.

    Returns
    -------
    pandas.DataFrame
        Table with the computed differences.  Empty if the required data is
        missing.
    """

    rad = summary[summary["STAGE"] == "radiating"]
    pre = summary[summary["STAGE"] == "pre"]
    post = summary[summary["STAGE"] == "post"]
    if rad.empty:
        return pd.DataFrame()

    min_d = float(rad["DOSE"].min())
    max_d = float(rad["DOSE"].max())

    os.makedirs(outdir, exist_ok=True)

    rows: list[dict[str, float]] = []
    diff_bias = [np.nan, np.nan]
    diff_dark = [np.nan, np.nan]

    for cal, diffs in (("BIAS", diff_bias), ("DARK", diff_dark)):
        r_first = rad[(rad["CALTYPE"] == cal) & (rad["DOSE"] == min_d)]
        r_last = rad[(rad["CALTYPE"] == cal) & (rad["DOSE"] == max_d)]
        p_pre = pre[pre["CALTYPE"] == cal]
        p_post = post[post["CALTYPE"] == cal]

        if not r_first.empty and not p_pre.empty:
            diff = float(r_first["MEAN"].mean() - p_pre["MEAN"].mean())
            diffs[0] = diff
            rows.append({"CALTYPE": cal, "CMP": "first_pre", "DOSE": min_d, "DIFF": diff})

        if not r_last.empty and not p_post.empty:
            diff = float(r_last["MEAN"].mean() - p_post["MEAN"].mean())
            diffs[1] = diff
            rows.append({"CALTYPE": cal, "CMP": "last_post", "DOSE": max_d, "DIFF": diff})

        valid = [d for d in diffs if np.isfinite(d)]
        doses = [min_d if np.isfinite(diffs[0]) else None, max_d if np.isfinite(diffs[1]) else None]
        plot_x = [d for d in doses if d is not None]
        plot_y = valid
        if plot_x:
            fig, ax = plt.subplots()
            ax.plot(plot_x, plot_y, "o-")
            ax.set_xlabel("Dose [kRad]")
            ax.set_ylabel("Base level difference [ADU]")
            ax.set_title(f"{cal} base level shift")
            ax.grid(True, ls="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"stage_base_diff_{cal.lower()}.png"))
            plt.close(fig)

    np.savez(
        os.path.join(outdir, "stage_base_diff.npz"),
        dose=np.array([min_d, max_d], dtype=float),
        bias_diff=np.array(diff_bias, dtype=float),
        dark_diff=np.array(diff_dark, dtype=float),
    )

    return pd.DataFrame(rows)


def _dynamic_range_analysis(summary: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Compute dynamic range and noise for each radiation dose.

    Two figures are produced: one showing the 16-bit dynamic range and its
    percentage reduction and another doing the same for the 12-bit range.  The
    mean dynamic range is plotted with a shaded band representing one standard
    deviation derived from the bias and dark standard deviations.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of the main processing loop with at least the ``STAGE``,
        ``CALTYPE``, ``DOSE``, ``MEAN`` and ``STD`` columns.
    outdir : str
        Destination directory for the plot and ``.npz`` arrays.

    Returns
    -------
    pandas.DataFrame
        Table with the computed statistics per dose.  Empty if the required
        data is missing.
    """

    df = summary[summary["STAGE"] == "during"]
    bias = df[df["CALTYPE"] == "BIAS"]
    dark = df[df["CALTYPE"] == "DARK"]
    doses = sorted(set(bias["DOSE"]) & set(dark["DOSE"]))

    rows: list[dict[str, float]] = []
    if not doses:
        return pd.DataFrame()

    os.makedirs(outdir, exist_ok=True)

    dr16_vals = []
    dr12_vals = []
    noise_vals = []
    noise_mag_vals = []
    red16_vals = []
    red12_vals = []
    base16_vals = []
    base12_vals = []

    for d in doses:
        b_rows = bias[bias["DOSE"] == d]
        d_rows = dark[dark["DOSE"] == d]
        bias_mean = float(b_rows["MEAN"].mean())
        dark_mean = float(d_rows["MEAN"].mean())
        bias_std = float(b_rows["STD"].mean())
        dark_std = float(d_rows["STD"].mean())

        base16 = bias_mean + dark_mean
        base12 = base16 / 16.0

        dr16 = 65536.0 - base16
        dr12 = 4096.0 - base12
        noise = float(np.sqrt(bias_std ** 2 + dark_std ** 2))
        noise_mag = float(1.0857 * noise / dr16) if dr16 > 0 else float("inf")

        dr16_vals.append(dr16)
        dr12_vals.append(dr12)
        noise_vals.append(noise)
        noise_mag_vals.append(noise_mag)
        red16_vals.append(100.0 * (65536.0 - dr16) / 65536.0)
        red12_vals.append(100.0 * (4096.0 - dr12) / 4096.0)
        base16_vals.append(base16)
        base12_vals.append(base12)

        rows.append(
            {
                "DOSE": float(d),
                "BIAS_MEAN": bias_mean,
                "DARK_MEAN": dark_mean,
                "DR_16": dr16,
                "DR_12": dr12,
                "NOISE_ADU": noise,
                "NOISE_MAG": noise_mag,
                "RED_16": red16_vals[-1],
                "RED_12": red12_vals[-1],
                "BASE_16": base16,
                "BASE_12": base12,
            }
        )

    # Save arrays for further inspection
    np.savez(
        os.path.join(outdir, "dynamic_range.npz"),
        dose=np.array(doses, dtype=float),
        bias_mean=np.array([r["BIAS_MEAN"] for r in rows], dtype=float),
        dark_mean=np.array([r["DARK_MEAN"] for r in rows], dtype=float),
        dynamic_range_16=np.array(dr16_vals, dtype=float),
        dynamic_range_12=np.array(dr12_vals, dtype=float),
        noise_adu=np.array(noise_vals, dtype=float),
        noise_mag=np.array(noise_mag_vals, dtype=float),
        reduction_16=np.array(red16_vals, dtype=float),
        reduction_12=np.array(red12_vals, dtype=float),
        base_level_16=np.array(base16_vals, dtype=float),
        base_level_12=np.array(base12_vals, dtype=float),
    )

    dr_err = np.array(noise_vals, dtype=float)

    # 16-bit dynamic range
    fig16, (ax16, ax16_red) = plt.subplots(2, 1, sharex=True)
    ax16.plot(doses, dr16_vals, "o-")
    ax16.fill_between(doses, np.array(dr16_vals) - dr_err, np.array(dr16_vals) + dr_err, alpha=0.2)
    ax16.axhline(65536, color="C2", ls="--", label="16-bit max")
    ax16.set_ylabel("Dynamic range [ADU]")
    ax16.set_title("16-bit dynamic range vs dose")
    ax16.grid(True, ls="--", alpha=0.5)
    ax16.legend()

    ax16_red.plot(doses, red16_vals, "o-", color="C1")
    ax16_red.set_xlabel("Dose [kRad]")
    ax16_red.set_ylabel("Reduction [%]")
    ax16_red.grid(True, ls="--", alpha=0.5)

    fig16.tight_layout()
    fig16.savefig(os.path.join(outdir, "dynamic_range_vs_dose_16.png"))
    plt.close(fig16)

    # 12-bit dynamic range
    fig12, (ax12, ax12_red) = plt.subplots(2, 1, sharex=True)
    ax12.plot(doses, dr12_vals, "s-")
    ax12.fill_between(doses, np.array(dr12_vals) - dr_err/16, np.array(dr12_vals) + dr_err/16, alpha=0.2)
    ax12.axhline(4096, color="C3", ls="--", label="12-bit max")
    ax12.set_ylabel("Dynamic range [ADU]")
    ax12.set_title("12-bit dynamic range vs dose")
    ax12.grid(True, ls="--", alpha=0.5)
    ax12.legend()

    ax12_red.plot(doses, red12_vals, "o-", color="C1")
    ax12_red.set_xlabel("Dose [kRad]")
    ax12_red.set_ylabel("Reduction [%]")
    ax12_red.grid(True, ls="--", alpha=0.5)

    fig12.tight_layout()
    fig12.savefig(os.path.join(outdir, "dynamic_range_vs_dose_12.png"))
    plt.close(fig12)

    # Baseline (bias + dark) vs dose in 16-bit units
    fig_b16, ax_b16 = plt.subplots()
    ax_b16.plot(doses, base16_vals, "o-", label="baseline")
    ax_b16.axhline(65536, color="C2", ls="--", label="max counts")
    ax_b16.set_xlabel("Dose [kRad]")
    ax_b16.set_ylabel("ADU (16 bit)")
    ax_b16.set_title("Baseline level vs dose (16-bit)")
    ax_b16.grid(True, ls="--", alpha=0.5)
    ax_b16.legend()
    fig_b16.tight_layout()
    fig_b16.savefig(os.path.join(outdir, "baseline_vs_dose_16.png"))
    plt.close(fig_b16)

    # Baseline vs dose in 12-bit units
    fig_b12, ax_b12 = plt.subplots()
    ax_b12.plot(doses, base12_vals, "o-", label="baseline")
    ax_b12.axhline(4096, color="C3", ls="--", label="max counts")
    ax_b12.set_xlabel("Dose [kRad]")
    ax_b12.set_ylabel("ADU (12 bit)")
    ax_b12.set_title("Baseline level vs dose (12-bit)")
    ax_b12.grid(True, ls="--", alpha=0.5)
    ax_b12.legend()
    fig_b12.tight_layout()
    fig_b12.savefig(os.path.join(outdir, "baseline_vs_dose_12.png"))
    plt.close(fig_b12)

    # Magnitude error and limit vs dose
    mag_lim_vals = -2.5 * np.log10(np.maximum(dr16_vals, 1e-9) / 65536.0)
    mag_lim_err = (2.5 / np.log(10)) * (np.array(noise_vals) / np.maximum(dr16_vals, 1e-9))

    fig_mag, (ax_err, ax_lim) = plt.subplots(2, 1, sharex=True)
    ax_err.plot(doses, noise_mag_vals, "o-")
    ax_err.set_ylabel("Mag error [mag]")
    ax_err.set_title("Magnitude error vs dose")
    ax_err.grid(True, ls="--", alpha=0.5)

    ax_lim.errorbar(doses, mag_lim_vals, yerr=mag_lim_err, fmt="o-")
    ax_lim.set_xlabel("Dose [kRad]")
    ax_lim.set_ylabel("Mag limit loss [mag]")
    ax_lim.set_title("Magnitude limit vs dose")
    ax_lim.grid(True, ls="--", alpha=0.5)

    fig_mag.tight_layout()
    fig_mag.savefig(os.path.join(outdir, "magnitude_vs_dose.png"))
    plt.close(fig_mag)

    return pd.DataFrame(rows)

def main(index_csv: str, output_dir: str) -> None:
    df = pd.read_csv(index_csv)
    df = df[df.get("BADFITS", False) == False]

    groups = _group_paths(df)

    master_dir = os.path.join(output_dir, "masters")
    os.makedirs(master_dir, exist_ok=True)

    records = []
    for (stage, cal, dose, exp), paths in tqdm(
        groups.items(), desc="Generating masters", unit="group"
    ):
        name = f"master_{cal.lower()}_{stage}_D{dose:g}kR_E{exp if exp is not None else 'none'}".replace('/', '_')
        fpath = os.path.join(master_dir, name + ".fits")

        if os.path.exists(fpath):
            master = fits.getdata(fpath)
            hdr = fits.getheader(fpath)
        else:
            master, hdr = _make_master(paths)
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

    pix_df = _pixel_precision_analysis(groups, os.path.join(output_dir, "pixel_precision"))
    _compare_stage_differences(summary, master_dir, os.path.join(output_dir, "analysis"))
    _fit_dose_response(summary, os.path.join(output_dir, "analysis"))
    _fit_base_level_trend(summary, os.path.join(output_dir, "analysis"))
    _stage_base_level_diff(summary, os.path.join(output_dir, "analysis"))
    _dynamic_range_analysis(summary, os.path.join(output_dir, "analysis"))

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
