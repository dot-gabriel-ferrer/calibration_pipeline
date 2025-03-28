# step8_evaluate_model.py
# Author: Elías Gabriel Ferrer Jorge

"""
Step 8: Evaluate the quality of the previously fitted dark model by comparing
synthetic dark frames with real data. This version uses the dark_mask_*.fits
files (already in ADU/s) rather than corrected dark frames.
It now uses the normalized model:

    DC(T) = A * exp(B * (T - t_min))

where t_min is loaded from the global parameters.
"""

import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from .step6_generate_synthetics import generate_precise_synthetic_dark, model_DC

def evaluate_model_quality(model_dir: str, dark_masks_dir: str, output_dir: str):
    """
    Evalúa la calidad del modelo ajustado comparando frames sintéticos con
    dark masks reales (en ADU/s). Cada mask debe tener en su header la
    palabra clave "TEMP" (o si no, se parsea del filename).

    El modelo normalizado es:
         DC(T) = A * exp(B * (T - t_min))
    cargándose t_min de los parámetros globales.

    Se computan errores (MAE, MAPE) y se generan gráficas de errores vs temperatura.

    :param model_dir: Directorio con A_map.fits, B_map.fits, hot_pixels.fits, global_params.npz.
    :param dark_masks_dir: Directorio con dark_mask_*.fits (cada uno en ADU/s).
    :param output_dir: Directorio donde se guardan los resultados (plots).
    """

    os.makedirs(output_dir, exist_ok=True)

    path_A_map = os.path.join(model_dir, "A_map.fits")
    path_B_map = os.path.join(model_dir, "B_map.fits")
    path_hotpix = os.path.join(model_dir, "hot_pixels.fits")
    path_global = os.path.join(model_dir, "global_params.npz")

    # Cargamos mapas y parámetros
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # para silenciar warnings de FITS
        with fits.open(path_A_map) as hdul:
            A_map = hdul[0].data.astype(np.float32)
        with fits.open(path_B_map) as hdul:
            B_map = hdul[0].data.astype(np.float32)
        with fits.open(path_hotpix) as hdul:
            hot_pixels = hdul[0].data.astype(bool)

    globals_ = np.load(path_global)
    A_g = float(globals_["A_g"])
    B_g = float(globals_["B_g"])
    t_min = float(globals_["t_min"])

    print(f"\nEvaluating model from '{model_dir}':")
    print(f"  A_map shape = {A_map.shape}, B_map shape = {B_map.shape}")
    print(f"  Global parameters: A_g={A_g:.4e}, B_g={B_g:.4e}, t_min={t_min:.2f}")

    mask_files = sorted(glob.glob(os.path.join(dark_masks_dir, "dark_mask_*.fits")))
    if not mask_files:
        print(f"No dark_mask_*.fits files found in '{dark_masks_dir}'. Aborting evaluation.")
        return

    results = []
    for fpath in mask_files:
        with fits.open(fpath) as hdul:
            hdr = hdul[0].header
            real_mask = hdul[0].data.astype(np.float32)

        temp = hdr.get("TEMP", None)
        if temp is None:
            basename = os.path.basename(fpath)
            try:
                temp = float(basename.replace("dark_mask_", "").replace(".fits", ""))
            except ValueError:
                print(f"Cannot determine temperature from file {basename}. Skipping.")
                continue

        synthetic = generate_precise_synthetic_dark(
            T_new=temp,
            A_map=A_map,
            B_map=B_map,
            hot_pixel_mask=hot_pixels,
            A_g=A_g,
            B_g=B_g,
            t_min=t_min,
            shape=real_mask.shape
        )

        diff = synthetic - real_mask
        mae = np.mean(np.abs(diff))
        mae_std = np.std(np.abs(diff))
        EPS = 1e-7
        mape = np.mean(np.abs(diff / (real_mask + EPS))) * 100
        mape_std = np.std(np.abs(diff / (real_mask + EPS))) * 100

        results.append((os.path.basename(fpath), temp, mae, mae_std, mape, mape_std))

    if not results:
        print("No valid masks found for evaluation.")
        return

    # Ordenamos resultados por temperatura
    results.sort(key=lambda x: x[1])
    _, temps_list, maes_list, maes_std_list, mapes_list, mapes_std_list = zip(*results)

    # Convertir a arrays
    temps_array = np.array(temps_list, dtype=np.float32)
    maes_array = np.array(maes_list, dtype=np.float32) #/ 16.0
    maes_std_array = np.array(maes_std_list, dtype=np.float32) #/ 16.0
    mapes_array = np.array(mapes_list, dtype=np.float32)
    mapes_std_array = np.array(mapes_std_list, dtype=np.float32)

    # Función interna para trazar en 2 ejes (MAE y MAPE)
    def _plot_evaluation(t_array, mae_array, mae_std_array, mape_array, mape_std_array,
                         title, out_fname):
        if len(t_array) == 0:
            print(f"Sin datos para graficar: {title}")
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_mae = 'tab:blue'
        color_mape = 'tab:green'

        ax1.plot(t_array, mae_array, 'o-', color=color_mae, label='MAE (ADU/s)')
        ax1.fill_between(t_array,
                         mae_array - mae_std_array,
                         mae_array + mae_std_array,
                         color=color_mae, alpha=0.1)
        ax1.set_xlabel('Temperatura (°C)')
        ax1.set_ylabel('MAE (ADU/s)', color=color_mae)
        ax1.tick_params(axis='y', labelcolor=color_mae)
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(t_array, mape_array, 'o--', color=color_mape, label='MAPE (%)')
        ax2.fill_between(t_array,
                         mape_array - mape_std_array,
                         mape_array + mape_std_array,
                         color=color_mape, alpha=0.1)
        ax2.set_ylabel('MAPE (%)', color=color_mape)
        ax2.tick_params(axis='y', labelcolor=color_mape)

        fig.suptitle(title)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        out_plot = os.path.join(output_dir, out_fname)
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Gráfico guardado en: {out_plot}")

    # 1) Gráfico con todos los datos
    _plot_evaluation(
        temps_array, maes_array, maes_std_array, mapes_array, mapes_std_array,
        "Evaluación del Modelo (Todos los datos)",
        "model_evaluation_masks_all.png"
    )

    # 2) Gráfico con MAPE < 100%
    mask_less_100 = (mapes_array < 100.0)
    _plot_evaluation(
        temps_array[mask_less_100],
        maes_array[mask_less_100],
        maes_std_array[mask_less_100],
        mapes_array[mask_less_100],
        mapes_std_array[mask_less_100],
        "Evaluación del Modelo (MAPE < 100%)",
        "model_evaluation_masks_mape_less_100.png"
    )

    # 3) Gráfico con 100% <= MAPE < 200%
    mask_100_200 = (mapes_array >= 100.0) & (mapes_array < 200.0)
    _plot_evaluation(
        temps_array[mask_100_200],
        maes_array[mask_100_200],
        maes_std_array[mask_100_200],
        mapes_array[mask_100_200],
        mapes_std_array[mask_100_200],
        "Evaluación del Modelo (100% <= MAPE < 200%)",
        "model_evaluation_masks_mape_100_200.png"
    )

    # 4) Gráfico con MAPE >= 200%
    mask_over_200 = (mapes_array >= 200.0)
    _plot_evaluation(
        temps_array[mask_over_200],
        maes_array[mask_over_200],
        maes_std_array[mask_over_200],
        mapes_array[mask_over_200],
        mapes_std_array[mask_over_200],
        "Evaluación del Modelo (MAPE >= 200%)",
        "model_evaluation_masks_mape_over_200.png"
    )

    print("Resumen de evaluación para dark masks:")
    for fname, t, mae, mae_std, mape, mape_std in results:
        print(f" - {fname} | T={t:.2f}°C => MAE={mae:.5f} ADU/s, MAPE={mape:.2f}%")


def evaluate_all_models(base_model_dir: str, base_mask_dir: str, base_output_dir: str):
    """
    Evalúa todos los modelos guardados en subdirectorios tipo exp_0p50 dentro de base_model_dir,
    comparando contra las dark masks en base_mask_dir/exp_0p50/, etc.

    Guarda los resultados en base_output_dir/exp_0p50/, etc.
    """
    for subdir in sorted(os.listdir(base_model_dir)):
        if not subdir.startswith("exp_"):
            continue

        model_dir = os.path.join(base_model_dir, subdir)
        mask_dir = os.path.join(base_mask_dir, subdir)
        output_dir = os.path.join(base_output_dir, subdir)

        if not os.path.isdir(model_dir) or not os.path.isdir(mask_dir):
            print(f"[!] Skipping {subdir}: Missing model or mask directory.")
            continue

        print(f"\n[📊] Evaluating model for {subdir}...")
        try:
            evaluate_model_quality(model_dir, mask_dir, output_dir)
        except Exception as e:
            print(f"[!] Error evaluating {subdir}: {e}")

            