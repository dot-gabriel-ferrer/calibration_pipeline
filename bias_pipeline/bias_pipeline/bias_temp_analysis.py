import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from steps.step1_load_data import load_bias_files
from steps.step2_generate_master_bias_by_temp import group_by_temperature
from steps.utils.utils_scaling import load_fits_scaled_12bit


def analyze_bias_by_temperature(basepath: str, output_dir: str) -> dict:
    """Compute mean statistics of bias frames grouped by temperature.

    Parameters
    ----------
    basepath : str
        Path to the directory containing bias FITS files (TestSection1).
    output_dir : str
        Folder where master bias frames and plots will be written.

    Returns
    -------
    dict
        Mapping of rounded temperature to statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    manager, bias_files = load_bias_files(basepath)
    if not bias_files:
        print("No bias files found.")
        return {}

    grouped = group_by_temperature(bias_files)
    stats = {}

    for temp in tqdm(sorted(grouped.keys()), desc="Processing temps", ncols=80):
        entries = grouped[temp]
        stack = []
        temps = []
        means = []
        for e in entries:
            data = load_fits_scaled_12bit(e["original_path"])
            stack.append(data)
            temps.append(e.get("temperature"))
            means.append(float(np.mean(data)))
        stack = np.stack(stack, axis=0)
        master = np.mean(stack, axis=0)
        fits.writeto(
            os.path.join(output_dir, f"master_bias_{temp:.1f}C.fits"),
            master.astype(np.float32),
            overwrite=True,
        )

        temp_mean = float(np.mean(temps))
        temp_std = float(np.std(temps))
        mean_adu = float(np.mean(means))
        std_adu = float(np.std(means))

        stats[temp] = {
            "temperature_mean": temp_mean,
            "temperature_std": temp_std,
            "mean_adu": mean_adu,
            "std_adu": std_adu,
            "values": means,
        }

        # Plot individual attempts per temperature
        x = np.arange(len(means))
        plt.figure()
        plt.plot(x, means, "o-")
        plt.hlines(mean_adu, 0, len(means) - 1, linestyles="--", color="red")
        plt.fill_between(
            x,
            mean_adu - std_adu,
            mean_adu + std_adu,
            color="red",
            alpha=0.3,
        )
        plt.title(f"T={temp_mean:.1f}±{temp_std:.1f}°C")
        plt.xlabel("Frame index")
        plt.ylabel("Mean (ADU)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"detail_T{temp:.1f}.png"))
        plt.close()

    # Global plot
    temps_sorted = sorted(stats.keys())
    temp_means = [stats[t]["temperature_mean"] for t in temps_sorted]
    mean_adu_list = [stats[t]["mean_adu"] for t in temps_sorted]
    std_adu_list = [stats[t]["std_adu"] for t in temps_sorted]

    plt.figure()
    plt.plot(temp_means, mean_adu_list, "o-", label="Mean ADU")
    plt.fill_between(
        temp_means,
        np.array(mean_adu_list) - np.array(std_adu_list),
        np.array(mean_adu_list) + np.array(std_adu_list),
        alpha=0.2,
        label="±1σ",
    )
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Mean (ADU)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_vs_temperature.png"))
    plt.close()

    with open(os.path.join(output_dir, "bias_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze bias frames grouped by temperature",
    )
    parser.add_argument("basepath", help="Path to TestSection1 bias directory")
    parser.add_argument(
        "--output-dir",
        default="bias_temp_analysis",
        help="Directory where results will be stored",
    )
    args = parser.parse_args()

    analyze_bias_by_temperature(args.basepath, args.output_dir)


if __name__ == "__main__":
    main()
