import os
import re
import glob
from typing import List, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from astropy.io import fits

from utils.raw_to_fits import convert_attempt


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
    return sorted(glob.glob(os.path.join(fits_dir, "*.fits")))


def _read_frame(path: str) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    return data, header


def _make_animation(frames: List[np.ndarray], times: List[float], rads: float, outpath: str) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="gray", origin="lower", animated=True)
    text = ax.text(0.02, 0.95, "", color="yellow", transform=ax.transAxes)

    def update(i):
        im.set_array(frames[i])
        text.set_text(f"t={times[i]:.1f}s\n{rads} kRads")
        return [im, text]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    ani.save(outpath, writer=PillowWriter(fps=8))
    plt.close(fig)


def _make_outlier_animation(frames: List[np.ndarray], times: List[float], outpath: str) -> None:
    ref = frames[0]
    diffs = [frame - ref for frame in frames]
    std = np.std(diffs[0]) if diffs else 0.0
    fig, ax = plt.subplots()
    im = ax.imshow(diffs[0], cmap="seismic", origin="lower", animated=True)
    scatter = ax.scatter([], [], facecolors="none", edgecolors="yellow", s=10)
    text = ax.text(0.02, 0.95, "", color="yellow", transform=ax.transAxes)

    def update(i):
        diff = diffs[i]
        im.set_array(diff)
        cur_std = np.std(diff)
        thresh = 5 * cur_std if cur_std > 0 else 0
        mask = np.abs(diff) > thresh
        coords = np.argwhere(mask)
        if coords.size:
            scatter.set_offsets(coords[:, [1, 0]])
        else:
            # scatter offsets expect an (N, 2) array; use shape (0, 2) when empty
            scatter.set_offsets(np.empty((0, 2)))
        text.set_text(f"t={times[i]:.1f}s")
        return [im, scatter, text]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    ani.save(outpath, writer=PillowWriter(fps=8))
    plt.close(fig)


def _plot_logs(rad_df: pd.DataFrame, power_df: pd.DataFrame, outpath: str) -> None:
    if rad_df.empty and power_df.empty:
        return
    time = _get_column(rad_df, ["TimeStamp", "Timestamp", "Time"])
    rad = _get_column(rad_df, ["RadiationLevel", "Dose"])
    temp = _get_column(rad_df, ["Temp", "Temperature"])
    amp = _get_column(power_df, ["Amperage", "Current"])
    volt = _get_column(power_df, ["Voltage"])

    plt.figure(figsize=(8, 6))
    idx = 1
    if rad is not None:
        plt.subplot(4, 1, idx)
        plt.plot(time, rad)
        plt.ylabel("Radiation")
        idx += 1
    if temp is not None:
        plt.subplot(4, 1, idx)
        plt.plot(time, temp)
        plt.ylabel("Temperature")
        idx += 1
    if amp is not None:
        plt.subplot(4, 1, idx)
        plt.plot(time, amp)
        plt.ylabel("Amperage")
        idx += 1
    if volt is not None:
        plt.subplot(4, 1, idx)
        plt.plot(time, volt)
        plt.ylabel("Voltage")
        idx += 1
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def analyze_directory(dir_path: str, output_dir: str) -> None:
    level = _parse_rads(os.path.basename(dir_path))
    if level is None:
        return

    rad_def = _load_csv(os.path.join(dir_path, "radiationLogDef.csv"))
    rad_log = _load_csv(os.path.join(dir_path, "radiationLog.csv"))
    power_log = _load_csv(os.path.join(dir_path, "powerLog.csv"))

    fits_paths = _load_frames(dir_path)
    data_list: List[np.ndarray] = []
    times: List[float] = []
    for fp in tqdm(fits_paths, desc="Loading frames", unit="frame"):
        data, hdr = _read_frame(fp)
        data_list.append(data)
        ts = hdr.get("TIMESTAMP")
        if ts is None:
            ts = hdr.get("TIME", 0.0)
        times.append(float(ts) - float(times[0]) if times else 0.0)

    if not data_list:
        return

    os.makedirs(output_dir, exist_ok=True)

    anim_path = os.path.join(output_dir, "frames.gif")
    _make_animation(data_list, times, level, anim_path)

    diff_anim_path = os.path.join(output_dir, "outliers.gif")
    _make_outlier_animation(data_list, times, diff_anim_path)

    plot_path = os.path.join(output_dir, "metrics.png")
    _plot_logs(rad_log, power_log, plot_path)

    vmin, vmax = np.percentile(data_list[0], [5, 95])
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(data_list[0], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("First")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(data_list[-1], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("Last")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "first_last.png"))
    plt.close()



def main(input_dir: str = "Operation", output_dir: str = "operation_results") -> None:
    os.makedirs(output_dir, exist_ok=True)
    entries = sorted(os.listdir(input_dir))
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
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
