# Author: Elías Gabriel Ferrer Jorge

"""
Step 7: Evaluate the flat model by comparing synthetic flats against real master flats.

1. Loads the fitted model maps (A, B, C) from step6.
2. Iterates over each real master flat (for various T, t_exp).
3. Generates a synthetic flat using the model and calculates:
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)
   - Global metrics (average, std)
4. Optionally saves results as FITS or PNG for visualization.

Assumes a linear model of the form:

   flat(T, t_exp) = A + B*(T - T_ref) + C*(t_exp - exp_ref)

(Adapt if your model is different.)
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_flat_model(model_dir: str):
    """
    Loads the parametric model maps: flat_A_map.fits, flat_B_map.fits, flat_C_map.fits.
    Returns them as 2D numpy arrays (A_map, B_map, C_map).
    """
    pathA = os.path.join(model_dir, "flat_A_map.fits")
    pathB = os.path.join(model_dir, "flat_B_map.fits")
    pathC = os.path.join(model_dir, "flat_C_map.fits")

    if not (os.path.exists(pathA) and os.path.exists(pathB) and os.path.exists(pathC)):
        raise FileNotFoundError("Missing one or more model maps in the specified directory.")

    A_map = fits.getdata(pathA).astype(np.float32)
    B_map = fits.getdata(pathB).astype(np.float32)
    C_map = fits.getdata(pathC).astype(np.float32)

    return A_map, B_map, C_map


def generate_synthetic_flat(A_map: np.ndarray, B_map: np.ndarray, C_map: np.ndarray,
                            T: float, t_exp: float, T_ref=0.0, exp_ref=0.0) -> np.ndarray:
    """
    Generates a synthetic flat given the model parameters and a target (T, t_exp).

    Model used:
       flat_ij(T, t_exp) = A_ij + B_ij*(T - T_ref) + C_ij*(t_exp - exp_ref)
    """
    return A_map + B_map*(T - T_ref) + C_map*(t_exp - exp_ref)


def evaluate_flat_model(master_dir: str, model_dir: str, output_dir: str,
                        T_ref=0.0, exp_ref=0.0, save_fits=False, save_plots=True):
    """
    Main function to evaluate the model by comparing synthetic vs. real master flats.

    :param master_dir: Directory with real master flats (from step5).
    :param model_dir: Directory with flat_A_map.fits, flat_B_map.fits, flat_C_map.fits.
    :param output_dir: Where to save evaluation outputs (FITS/plots).
    :param T_ref: Reference temperature used when fitting the model.
    :param exp_ref: Reference exposure used when fitting the model.
    :param save_fits: Whether to save difference, MAE, MAPE as FITS.
    :param save_plots: Whether to generate PNG images of difference, MAE, MAPE.
    """
    print("\n[Step 7] Evaluating Flat Model against real master flats...")

    os.makedirs(output_dir, exist_ok=True)
    A_map, B_map, C_map = load_flat_model(model_dir)

    fits_files = [f for f in os.listdir(master_dir) if f.lower().endswith(".fits")]
    results = []

    for fname in tqdm(fits_files, desc="Evaluating", ncols=80):
        path = os.path.join(master_dir, fname)
        with fits.open(path) as hdul:
            real_data = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header

        # Attempt to parse T, exp from filename or header
        # e.g. "master_flat_R_T10.0_E60.0.fits"
        parts = fname.split("_")
        if len(parts) >= 4:
            Tpart = parts[3]  # T10.0
            Epart = parts[4]  # E60.0.fits
        else:
            # fallback
            Tpart = f"T{hdr.get('T_AVG', 0.0):.1f}"
            Epart = f"E{hdr.get('E_AVG', 0.0):.1f}"

        # parse
        try:
            temp_val = float(Tpart.replace("T","").replace(".fits",""))
        except:
            temp_val = hdr.get('T_AVG', 0.0)
        try:
            exp_val = float(Epart.replace("E","").replace(".fits",""))
        except:
            exp_val = hdr.get('E_AVG', 0.0)

        # Generate synthetic
        synthetic = generate_synthetic_flat(A_map, B_map, C_map,
                                            T=temp_val, t_exp=exp_val,
                                            T_ref=T_ref, exp_ref=exp_ref)

        # Compute metrics
        diff = real_data - synthetic
        mae = np.abs(diff)
        # Avoid division by zero
        denom = np.where(real_data == 0.0, 1.0, real_data)
        mape = np.abs(diff / denom) * 100.0

        mae_mean = float(np.nanmean(mae))
        mape_mean = float(np.nanmean(mape))

        results.append({
            'filename': fname,
            'temperature': temp_val,
            'exposure': exp_val,
            'mae': mae_mean,
            'mape': mape_mean
        })

        # Save FITS if requested
        if save_fits:
            diff_path  = os.path.join(output_dir, fname.replace(".fits","_diff.fits"))
            mae_path   = os.path.join(output_dir, fname.replace(".fits","_mae.fits"))
            mape_path  = os.path.join(output_dir, fname.replace(".fits","_mape.fits"))

            fits.writeto(diff_path, diff.astype(np.float32), overwrite=True)
            fits.writeto(mae_path, mae.astype(np.float32), overwrite=True)
            fits.writeto(mape_path, mape.astype(np.float32), overwrite=True)

        # Save plots if requested
        if save_plots:
            import matplotlib.pyplot as plt
            def plot_and_save(img, title, out_name, cmap='viridis', vmax=None):
                plt.figure(figsize=(5,4))
                plt.imshow(img, cmap=cmap, vmax=vmax)
                plt.title(title)
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, out_name), dpi=120)
                plt.close()

            # Diff
            plot_and_save(diff, f"Diff (Real-Synth) T={temp_val:.1f}, Exp={exp_val:.1f}",
                          fname.replace(".fits","_diff.png"), cmap='RdBu', vmax=None)
            # MAE
            plot_and_save(mae, f"MAE T={temp_val:.1f}, Exp={exp_val:.1f}",
                          fname.replace(".fits","_mae.png"), cmap='hot', vmax=None)
            # MAPE
            plot_and_save(mape, f"MAPE T={temp_val:.1f}, Exp={exp_val:.1f}",
                          fname.replace(".fits","_mape.png"), cmap='hot', vmax=100)

    # Print summary
    print("\n[Step 7] Summary of evaluation results:")
    for r in results:
        print(f" - {r['filename']}: T={r['temperature']} Exp={r['exposure']} -> MAE={r['mae']:.4f}, MAPE={r['mape']:.2f}%")

    # Optionally save a CSV or JSON
    # e.g.:
    # import pandas as pd
    # df = pd.DataFrame(results)
    # df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)

    print("\n[Step 7] Evaluation complete.")
