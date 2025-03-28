# step9_evaluate_dark_current.py
# Author: Elías Gabriel Ferrer Jorge

"""
Step 9: Evaluate dark current per temperature.

This module provides a function that loads each dark mask (already in ADU/s)
from a specified directory, extracts the temperature, computes the mean dark current,
and generates a plot of mean dark current versus temperature.
The resulting plot is saved to an output directory.
"""

import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def evaluate_dark_current_by_temperature(dark_masks_dir: str, output_dir: str):
    """
    Evaluates the mean dark current for each temperature by reading dark mask FITS files.

    The function:
      1. Loads each FITS file in `dark_masks_dir` matching 'dark_mask_*.fits'.
      2. Extracts the temperature from the FITS header (keyword "TEMP") or, if not available,
         parses it from the filename.
      3. Computes the mean dark current (in ADU/s) of the dark mask.
      4. Generates a plot of mean dark current versus temperature.
      5. Saves the plot in `output_dir` as 'dark_current_vs_temperature.png'.

    :param dark_masks_dir: Directory containing dark mask FITS files (each in ADU/s).
    :type dark_masks_dir: str
    :param output_dir: Directory where the resulting plot will be saved.
    :type output_dir: str
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(dark_masks_dir, "dark_mask_*.fits")))
    if not mask_files:
        print("No dark mask files found in the specified directory.")
        return

    temperatures = []
    mean_dark_currents = []
    std_dark_currents = []
    for fpath in mask_files:
        with fits.open(fpath) as hdul:
            header = hdul[0].header
            data = hdul[0].data.astype(np.float32)
        temp = header.get("TEMP", None)
        if temp is None:
            # Si no se encuentra la variable TEMP, intentamos parsearla del nombre del archivo
            basename = os.path.basename(fpath)
            try:
                temp = float(basename.replace("dark_mask_", "").replace(".fits", ""))
            except ValueError:
                print(f"Could not determine temperature from file {basename}")
                continue
        mean_current = np.mean(data) #/ 16
        std_current = np.std(data) #/ 16
        temperatures.append(temp)
        mean_dark_currents.append(mean_current)
        std_dark_currents.append(std_current)

    # Convertir a arrays y ordenar por temperatura
    temperatures = np.array(temperatures)
    mean_dark_currents = np.array(mean_dark_currents)
    std_dark_currents = np.array(std_dark_currents)

    sort_idx = np.argsort(temperatures)
    
    temperatures_sorted = temperatures[sort_idx]
    currents_sorted = mean_dark_currents[sort_idx]
    currents_std_sorted = std_dark_currents[sort_idx]

    # Generar el plot
    
    color_mae = 'tab:blue'
    color_mape = 'tab:green'


    plt.figure(figsize=(8, 6))
    plt.plot(temperatures_sorted, currents_sorted, 'o-', color=color_mae, label="Mean Dark Current")
    plt.fill_between(temperatures_sorted,
                     currents_sorted - currents_std_sorted,
                     currents_sorted + currents_std_sorted,
                     color=color_mae, alpha=0.1)
    plt.title("Dark Current vs Temperature in 12 bits scaled from 16 bits")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Mean Dark Current (ADU/s)")
    plt.grid(True)
    plt.legend()
    output_path = os.path.join(output_dir, "dark_current_vs_temperature.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Dark current evaluation plot saved at {output_path}")

def evaluate_dark_current_all_exposures(base_mask_dir: str, base_output_dir: str):
    """
    Ejecuta la evaluación del dark current promedio vs temperatura para
    cada grupo de dark masks organizadas por exposición.

    Se asume la estructura:
        base_mask_dir/
            exp_0p50/
                dark_mask_*.fits
            exp_1p00/
            ...

    Y guarda los resultados en:
        base_output_dir/
            exp_0p50/
                dark_current_vs_temperature.png
            ...
    """
    for subdir in sorted(os.listdir(base_mask_dir)):
        if not subdir.startswith("exp_"):
            continue

        mask_dir = os.path.join(base_mask_dir, subdir)
        output_dir = os.path.join(base_output_dir, subdir)

        if not os.path.isdir(mask_dir):
            print(f"[!] Skipping {subdir}: not a directory.")
            continue

        print(f"\n[📈] Evaluating mean dark current for {subdir}...")
        try:
            evaluate_dark_current_by_temperature(mask_dir, output_dir)
        except Exception as e:
            print(f"[!] Error in {subdir}: {e}")