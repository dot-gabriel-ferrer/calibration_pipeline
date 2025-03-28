# step11_evaluate_model_2D.py
# Author: Elías Gabriel Ferrer Jorge

"""
Evaluate the 2D dark current model:
DC(T, t_exp) = A * t_exp^gamma * exp(B * (T - t_min))

Compares the synthetic frame against real dark current masks for
each (temperature, exposure) pair.

Generates:
- Scatter plot: real vs synthetic
- Pixel-by-pixel comparison plot
- Heatmaps of MAE, MAPE
- Heatmaps of real and synthetic dark current
- Summary CSV
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pandas as pd


def extract_temp_exp(filename: str):
    pattern = r"t([-\d.]+)_e([-\d.]+)"
    match = re.search(pattern, filename)
    if match:
        try:
            temp_str = match.group(1).rstrip(".")
            exp_str = match.group(2).rstrip(".")
            temp = float(temp_str)
            exp = float(exp_str)
            return temp, exp
        except ValueError:
            return None, None
    return None, None


def generate_synthetic_frame_2D(T, t_exp, A_map, B_map, GAMMA_map, t_min, shape):
    with np.errstate(divide="ignore", invalid="ignore"):
        synth = A_map * (t_exp ** GAMMA_map) * np.exp(B_map * (T - t_min))
    return np.nan_to_num(synth, nan=0.0, posinf=0.0, neginf=0.0)


def plot_heatmap(data2D, x_labels, y_labels, title, cmap, out_path, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data2D, cmap=cmap, aspect='auto', origin='lower')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels([f"{x:.1f}" for x in x_labels])
    ax.set_yticklabels([f"{y:.1f}" for y in y_labels])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Exposure [s]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_model_2D(model_dir: str, masks_root: str, output_dir: str):
    print("[•] Evaluando modelo 2D...")

    A_map = fits.getdata(os.path.join(model_dir, "A_map_2D.fits"))
    B_map = fits.getdata(os.path.join(model_dir, "B_map_2D.fits"))
    GAMMA_map = fits.getdata(os.path.join(model_dir, "GAMMA_map_2D.fits"))
    t_min = np.load(os.path.join(model_dir, "global_params_2d.npz"))["t_min"].item()
    shape = A_map.shape

    results = {}
    comparison_dir = os.path.join(output_dir, "plots_2d_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(masks_root, "exp_*/dark_mask_t*_e*.fits")))

    temps_unique = set()
    exps_unique = set()

    # Dicts para heatmaps
    mae_map = {}
    mape_map = {}
    real_dc_map = {}
    synth_dc_map = {}

    for f in tqdm(all_files, desc="Evaluando máscaras reales"):
        temp, exp = extract_temp_exp(os.path.basename(f))
        if temp is None or exp is None:
            continue

        temps_unique.add(temp)
        exps_unique.add(exp)

        real = fits.getdata(f).astype(np.float32)
        synth = generate_synthetic_frame_2D(temp, exp, A_map, B_map, GAMMA_map, t_min, shape)

        tag = f"T{temp:.1f}_E{exp:.1f}".replace(".", "p")

        # Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(real.flatten(), synth.flatten(), s=1, alpha=0.3)
        max_val = max(np.max(real), np.max(synth))
        plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal y=x")
        plt.xlabel("Real [ADU/s]")
        plt.ylabel("Synthetic [ADU/s]")
        plt.title(f"Scatter Real vs Synthetic\nT={temp:.1f}°C, t={exp:.1f}s")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f"scatter_{tag}.png"), dpi=150)
        plt.close()

        # Comparación ordenada
        sorted_real = np.sort(real.flatten())
        sorted_synth = np.sort(synth.flatten())
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_real, label="Real", linewidth=1)
        plt.plot(sorted_synth, label="Synthetic", linewidth=1)
        plt.xlabel("Pixel index (sorted)")
        plt.ylabel("Dark Current [ADU/s]")
        plt.title(f"Pixel-wise Comparison\nT={temp:.1f}°C, t={exp:.1f}s")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f"line_compare_{tag}.png"), dpi=150)
        plt.close()

        # Métricas
        mae = np.mean(np.abs(synth - real))
        mape = np.mean(np.abs((synth - real) / (real + 1e-6))) * 100

        results[(temp, exp)] = (mae, mape)

        mae_map[(temp, exp)] = mae
        mape_map[(temp, exp)] = mape
        real_dc_map[(temp, exp)] = np.mean(real)
        synth_dc_map[(temp, exp)] = np.mean(synth)

    # DataFrame resumen
    rows = []
    for (t, e), (mae, mape) in sorted(results.items()):
        rows.append({
            "Temperature (°C)": t,
            "Exposure (s)": e,
            "MAE [ADU/s]": mae,
            "MAPE [%]": mape
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "evaluation_model_2d_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Resultados guardados en {csv_path}")

    # Crear heatmaps
    temps_sorted = sorted(list(temps_unique))
    exps_sorted = sorted(list(exps_unique))

    def map_to_2d(data_dict):
        data = np.full((len(exps_sorted), len(temps_sorted)), np.nan)
        for i, e in enumerate(exps_sorted):
            for j, t in enumerate(temps_sorted):
                if (t, e) in data_dict:
                    data[i, j] = data_dict[(t, e)]
        return data

    mae_array = map_to_2d(mae_map)
    mape_array = map_to_2d(mape_map)
    real_array = map_to_2d(real_dc_map)
    synth_array = map_to_2d(synth_dc_map)

    # Graficar heatmaps
    plot_heatmap(mae_array, temps_sorted, exps_sorted, "MAE [ADU/s]", "viridis",
                os.path.join(output_dir, "heatmap_mae_2d.png"), "MAE [ADU/s]")
    plot_heatmap(mape_array, temps_sorted, exps_sorted, "MAPE [%]", "plasma",
                os.path.join(output_dir, "heatmap_mape_2d.png"), "MAPE [%]")
    plot_heatmap(real_array, temps_sorted, exps_sorted, "Real Dark Current [ADU/s]", "cividis",
                os.path.join(output_dir, "heatmap_dark_real_2d.png"), "Real DC [ADU/s]")
    plot_heatmap(synth_array, temps_sorted, exps_sorted, "Synthetic Dark Current [ADU/s]", "cividis",
                os.path.join(output_dir, "heatmap_dark_synthetic_2d.png"), "Synthetic DC [ADU/s]")