import os
import re
import glob
from typing import List, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from astropy.io import fits

from utils.raw_to_fits import convert_attempt, parse_frame_number

"""Analyse radiation operation sequences.

The script converts raw frames to FITS if necessary, detects per-frame outliers
and correlates them with radiation logs.  Output figures include an overall
metric plot, intensity trends and the new ``rad_vs_outliers.png`` which shows
the relationship between radiation dose or level and detected outliers.
"""

def _parse_rads(dirname: str) -> Optional[float]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)kRads", dirname)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _get_column(df: pd.DataFrame, patterns: List[str]) -> Optional[pd.Series]:
    for pat in patterns:
        for col in df.columns:
            if re.search(pat, col, re.IGNORECASE):
                return df[col]
    return None


def _load_frames(attempt_dir: str) -> List[str]:
    fits_dir = os.path.join(attempt_dir, "fits")
    if not os.path.isdir(fits_dir) or not glob.glob(os.path.join(fits_dir, "*.fits")):
        convert_attempt(attempt_dir, calibration="OPER")

    paths = glob.glob(os.path.join(fits_dir, "*.fits"))

    def sort_key(path: str) -> float:
        num = parse_frame_number(os.path.basename(path))
        hdr = None
        if num is None:
            try:
                hdr = fits.getheader(path)
            except Exception:
                hdr = None
            if hdr is not None:
                try:
                    num = int(hdr.get("FRAMENUM")) if hdr.get("FRAMENUM") is not None else None
                except Exception:
                    num = None
        if num is not None:
            return float(num)

        if hdr is None:
            try:
                hdr = fits.getheader(path)
            except Exception:
                return float("inf")
        ts = hdr.get("TIMESTAMP")
        if ts is None:
            ts = hdr.get("TIME")
        try:
            return float(ts)
        except Exception:
            return float("inf")

    return sorted(paths, key=sort_key)


def _read_frame(path: str) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    return data, header


def _detect_outliers(frames: List[np.ndarray], sigma: float = 5) -> tuple[list[np.ndarray], np.ndarray]:
    """Return per-frame boolean masks of outlier pixels and a cumulative mask."""
    if not frames:
        return [], np.array([])

    stack = np.stack(frames)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)

    masks = [np.abs(frame - mean) > sigma * std for frame in frames]
    cumulative = np.any(np.stack(masks), axis=0)
    return masks, cumulative


def _make_animation(frames: List[np.ndarray], times: List[float], rads: float, outpath: str) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="gray", origin="lower", animated=True)
    text = ax.text(0.02, 1.05, "", transform=ax.transAxes, color="#006400")

    def update(i):
        im.set_array(frames[i])
        text.set_text(f"t={times[i]:.1f}s\n{rads} kRads")
        return [im, text]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    ani.save(outpath, writer=PillowWriter(fps=8))
    mp4_path = os.path.splitext(outpath)[0] + ".mp4"
    if shutil.which("ffmpeg"):
        try:
            ani.save(mp4_path, writer=FFMpegWriter(fps=8))
        except Exception:
            pass
    plt.close(fig)


def _make_outlier_animation(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    times: List[float],
    outpath: str,
) -> None:
    stack = np.stack(frames)
    global_mean = np.mean(stack)
    global_std = np.std(stack)
    vmin = global_mean - 5 * global_std
    vmax = global_mean + 5 * global_std

    masked_frames = [np.where(m, np.clip(f, vmin, vmax), 0) for f, m in zip(frames, masks)]

    fig, ax = plt.subplots()
    im = ax.imshow(masked_frames[0], cmap="gray", origin="lower", animated=True, vmin=vmin, vmax=vmax)
    text = ax.text(0.02, 1.05, "", transform=ax.transAxes, color="#006400")

    def update(i: int):
        im.set_array(masked_frames[i])
        text.set_text(f"t={times[i]:.1f}s\nOutliers={int(np.sum(masks[i]))}")
        return [im, text]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    ani.save(outpath, writer=PillowWriter(fps=8))
    mp4_path = os.path.splitext(outpath)[0] + ".mp4"
    if shutil.which("ffmpeg"):
        try:
            ani.save(mp4_path, writer=FFMpegWriter(fps=8))
        except Exception:
            pass
    plt.close(fig)


def _plot_logs(
    rad_df: pd.DataFrame,
    power_df: pd.DataFrame,
    outpath: str,
    frame_times: list[float] | None = None,
    outlier_counts: list[int] | None = None,
) -> None:
    if rad_df.empty and power_df.empty and outlier_counts is None:
        return

    time = _get_column(rad_df, ["TimeStamp", "Timestamp", "Time"])
    if time is None:
        time = _get_column(power_df, ["TimeStamp", "Timestamp", "Time", "TimeStampPwr"])
    if time is not None:
        time = pd.to_numeric(time, errors="coerce")
        time = time - float(time.iloc[0])

    rad = _get_column(rad_df, ["Dose"])  # commanded level
    rad_level = _get_column(rad_df, ["RadiationLevel"])
    temp = _get_column(rad_df, ["Temp", "Temperature"])
    amp = _get_column(power_df, ["Amperage", "Current"])
    volt = _get_column(power_df, ["Voltage"])

    rows = sum(
        x is not None
        for x in (rad, rad_level, temp, amp, volt)
    )
    if outlier_counts is not None and frame_times is not None:
        rows += 1
    if rows == 0:
        return
    plt.figure(figsize=(8, 1.5 * rows))
    idx = 1

    if rad is not None:
        plt.subplot(rows, 1, idx)

        plt.plot(time, rad)
        plt.ylabel("Radiation")
        idx += 1

    if rad_level is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(time, rad_level)
        plt.ylabel("Sensor Rad")
        idx += 1

    if temp is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(time, temp)
        plt.ylabel("Temperature")
        idx += 1

    if amp is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(time, amp)
        plt.ylabel("Amperage")
        idx += 1

    if volt is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(time, volt)
        plt.ylabel("Voltage")
        idx += 1

    if outlier_counts is not None and frame_times is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(frame_times, outlier_counts)
        plt.ylabel("Outlier Count")
        idx += 1

    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _plot_intensity_stats(
    means: list[float], stds: list[float], times: list[float], outpath: str
) -> None:
    if not means:
        return

    t = np.array(times, dtype=float)
    m = np.array(means, dtype=float)
    s = np.array(stds, dtype=float)

    if len(t) > 1:
        try:
            mean_fit = np.polyfit(t, m, 1)
        except np.linalg.LinAlgError:
            mean_fit = (0.0, float(np.mean(m)))
        try:
            std_fit = np.polyfit(t, s, 1)
        except np.linalg.LinAlgError:
            std_fit = (0.0, float(np.mean(s)))
    else:
        mean_fit = (0.0, float(np.mean(m)))
        std_fit = (0.0, float(np.mean(s)))
    trend_t = np.array([t[0], t[-1]])

    plt.figure(figsize=(8, 4))
    plt.plot(t, m, label="Mean")
    plt.plot(t, s, label="StdDev")
    plt.plot(trend_t, mean_fit[0] * trend_t + mean_fit[1], "k--", label="Mean Trend")
    plt.plot(trend_t, std_fit[0] * trend_t + std_fit[1], "r--", label="Std Trend")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _plot_rad_vs_outliers(
    rad_df: pd.DataFrame,
    frame_times: list[float],
    outlier_counts: list[int],
    outpath: str,
) -> None:
    """Plot radiation dose/level versus outlier count and save the figure."""
    if rad_df.empty or not frame_times:
        return

    # radiation measurement (commanded dose or sensor level)
    rad = _get_column(rad_df, ["Dose"]) or _get_column(rad_df, ["RadiationLevel"])
    if rad is None:
        return

    time = _get_column(rad_df, ["TimeStamp", "Timestamp", "Time"])
    if time is None:
        return

    rad = pd.to_numeric(rad, errors="coerce")
    time = pd.to_numeric(time, errors="coerce")
    rad_time = time - float(time.iloc[0])

    # drop NaNs before interpolation
    valid = ~(rad.isna() | rad_time.isna())
    rad_time = rad_time[valid]
    rad = rad[valid]

    if rad.empty:
        return

    rad_values = np.interp(frame_times, rad_time, rad)

    corr = np.corrcoef(rad_values, outlier_counts)[0, 1] if len(frame_times) > 1 else np.nan
    fit = np.polyfit(rad_values, outlier_counts, 1) if len(frame_times) > 1 else (0.0, outlier_counts[0])
    line_x = np.array([rad_values.min(), rad_values.max()])
    line_y = fit[0] * line_x + fit[1]

    plt.figure()
    plt.scatter(rad_values, outlier_counts, s=10, label="Frames")
    plt.plot(line_x, line_y, "r--", label="Linear fit")
    if not np.isnan(corr):
        plt.title(f"Correlation r={corr:.2f}")
    plt.xlabel("Radiation")
    plt.ylabel("Outlier Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def analyze_directory(dir_path: str, output_dir: str) -> None:
    level = _parse_rads(os.path.basename(dir_path))
    if level is None:
        return

    rad_def = _load_csv(os.path.join(dir_path, "radiationLogDef.csv"))
    rad_log = _load_csv(os.path.join(dir_path, "radiationLog.csv"))
    if rad_log.empty:
        rad_log = _load_csv(os.path.join(dir_path, "radiationLogCompleto.csv"))
    power_log = _load_csv(os.path.join(dir_path, "powerLog.csv"))

    fits_paths = _load_frames(dir_path)
    data_list: List[np.ndarray] = []
    times: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    for fp in tqdm(fits_paths, desc="Loading frames", unit="frame"):
        data, hdr = _read_frame(fp)
        data_list.append(data)
        means.append(float(np.mean(data)))
        stds.append(float(np.std(data)))
        ts = hdr.get("TIMESTAMP")
        if ts is None:
            ts = hdr.get("TIME", 0.0)
        times.append(float(ts) - float(times[0]) if times else 0.0)

    if not data_list:
        return

    mean_frame = np.mean(np.stack(data_list), axis=0)

    os.makedirs(output_dir, exist_ok=True)

    anim_path = os.path.join(output_dir, "frames.gif")
    _make_animation(data_list, times, level, anim_path)

    # detect outliers across the sequence
    masks, cumulative = _detect_outliers(data_list, sigma=5)

    diff_anim_path = os.path.join(output_dir, "outliers.gif")
    _make_outlier_animation(data_list, masks, times, diff_anim_path)

    # compute outlier counts per frame using masks
    outlier_counts = [int(np.sum(m)) for m in masks]

    # save cumulative mask
    plt.imsave(os.path.join(output_dir, "cumulative_outliers.png"), cumulative, cmap="gray", origin="lower")
    fits.writeto(
        os.path.join(output_dir, "cumulative_outliers.fits"),
        cumulative.astype(np.uint8),
        overwrite=True,
    )

    plot_path = os.path.join(output_dir, "metrics.png")
    _plot_logs(rad_log, power_log, plot_path, times, outlier_counts)

    rad_out_path = os.path.join(output_dir, "rad_vs_outliers.png")
    _plot_rad_vs_outliers(rad_log, times, outlier_counts, rad_out_path)

    intens_path = os.path.join(output_dir, "intensity_stats.png")
    _plot_intensity_stats(means, stds, times, intens_path)

    vmin, vmax = np.percentile(data_list[0], [5, 95])
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(data_list[0], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("First")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(mean_frame, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("Mean")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(data_list[-1], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("Last")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "first_last.png"))
    plt.close()



def main(
    input_dir: str = "Operation",
    output_dir: str = "operation_results",
    datasets: Optional[list[str]] = None,
) -> None:
    """Analyze one or more operation datasets.

    Parameters
    ----------
    input_dir : str, optional
        Directory containing the dataset sub-directories.
    output_dir : str, optional
        Destination directory for the analysis results.
    datasets : list[str] or None, optional
        Specific dataset sub-directories to process.  When ``None`` (default)
        all ``*kRads`` folders inside ``input_dir`` are analyzed.
    """

    os.makedirs(output_dir, exist_ok=True)
    entries = sorted(os.listdir(input_dir))
    if datasets:
        entries = [e for e in entries if e in datasets]
    for entry in tqdm(entries, desc="Directories", unit="dir"):
        dpath = os.path.join(input_dir, entry)
        if os.path.isdir(dpath) and _parse_rads(entry) is not None:
            analyze_directory(dpath, os.path.join(output_dir, entry))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze radiation operation frames")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="Operation",
        help="Directory containing Operation attempts",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="operation_results",
        help="Directory where analysis outputs will be written",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Names of subdirectories inside input_dir to analyze. If omitted, "
            "all <x>kRads folders are processed."
        ),
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.datasets)
