# Author: Elías Gabriel Ferrer Jorge

"""
Step 6: Fit a parametric flat model based on the generated master flats.

Two possible approaches (simplified examples):
  1) Global model: One set of parameters for the entire sensor (no per-pixel).
  2) Pixel-wise model: Each pixel has its own set of parameters, analogous to dark/bias modeling.

This script demonstrates a pixel-wise linear approach:
    flat_ij(T, t_exp) = A_ij + B_ij*(T - T_ref) + C_ij*(t_exp - t_ref)

(You can adapt to other forms, e.g. polynomials, exponentials, etc.)
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm


def load_master_flats(master_dir: str):
    """
    Loads all master flat FITS from a directory. 
    Expects filenames that encode filter, temperature, exposure in some manner.

    :param master_dir: Path to directory of master flats (from step5).
    :return: A dict {(filter, temp, exp): 2D np.array}, plus shape info for reference
    """
    master_map = {}
    fits_files = [f for f in os.listdir(master_dir) if f.lower().endswith(".fits")]

    for fname in fits_files:
        # parse filter, T, exp from filename or from header
        # For example: "master_flat_R_T10.0_E60.0.fits"
        # We'll do a naive parse:
        path = os.path.join(master_dir, fname)
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
        # parse from filename
        # you can do something more robust if your naming scheme is different
        parts = fname.split("_")
        if len(parts) >= 4:
            filter_val = parts[2]  # e.g. "R"
            Tpart = parts[3]       # e.g. "T10.0"
            Epart = parts[4]       # e.g. "E60.0.fits"
        else:
            filter_val = hdr.get("F_SAMP", "UNKNOWN")  # fallback to header
            Tpart = f"T{hdr.get('T_AVG',0):.1f}"
            Epart = f"E{hdr.get('E_AVG',0):.1f}"

        try:
            temp_str = Tpart.replace("T", "").replace(".fits", "")
            temp_val = float(temp_str)
        except ValueError:
            temp_val = hdr.get("T_AVG", 0.0)

        try:
            exp_str = Epart.replace("E", "").replace(".fits", "")
            exp_val = float(exp_str)
        except ValueError:
            exp_val = hdr.get("E_AVG", 0.0)

        key = (filter_val, temp_val, exp_val)
        master_map[key] = data

    return master_map


def fit_pixelwise_linear_flat_model(master_map, T_ref=0.0, exp_ref=0.0):
    """
    Example pixel-wise linear model:

        flat_ij(T, t_exp) = A_ij + B_ij*(T - T_ref) + C_ij*(t_exp - t_ref)

    1) We'll gather all (T, t_exp) combos and stack the corresponding master flats.
    2) Solve linear regression per pixel using least squares.
    3) Return A_map, B_map, C_map.

    :param master_map: dict {(filter, T, exp): 2D array of master flat}
                       We'll consider only ONE filter for simplicity here,
                       or you can extend to multi-filter if needed.
    :param T_ref: reference temperature
    :param exp_ref: reference exposure
    :return: (A_map, B_map, C_map)
    """
    # Suppose we do it for only one filter. 
    # If you have multiple filters, you'd either do separate fits or add filter as dimension.
    # Let's pick the first filter found:
    # (filter_val, T, exp) in master_map
    unique_filters = {k[0] for k in master_map.keys()}
    if len(unique_filters) != 1:
        print("[Warning] More than one filter found. Using the first one for demonstration.")
    chosen_filter = list(unique_filters)[0]

    # Gather entries
    entries = [(k[1], k[2]) for k in master_map.keys() if k[0] == chosen_filter]
    temps = []
    exps = []
    data_stack = []
    for t, e in entries:
        temps.append(t)
        exps.append(e)
        data_stack.append(master_map[(chosen_filter, t, e)])

    data_stack = np.stack(data_stack, axis=0)  # shape (n, H, W)
    temps = np.array(temps)
    exps = np.array(exps)
    n, H, W = data_stack.shape

    # Build design matrix X with shape (n, 3):
    # [1, (T - T_ref), (exp - exp_ref)]
    X = np.column_stack([
        np.ones(n),
        temps - T_ref,
        exps - exp_ref
    ])  # shape (n,3)

    # Flatten Y to shape (n, H*W)
    Y = data_stack.reshape(n, -1)  # (n, H*W)

    # Solve for Beta in: Y ~ X * Beta
    # Beta shape: (3, H*W)
    # We'll do a simple least squares: Beta = (X^T X)^-1 X^T Y
    # but for performance, we can use np.linalg.lstsq
    Beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # shape (3, H*W)

    A_map = Beta[0].reshape(H, W)
    B_map = Beta[1].reshape(H, W)
    C_map = Beta[2].reshape(H, W)

    return A_map.astype(np.float32), B_map.astype(np.float32), C_map.astype(np.float32)


def save_flat_model(A_map, B_map, C_map, output_dir):
    """
    Saves the parametric model as FITS files: flat_A_map.fits, flat_B_map.fits, flat_C_map.fits
    """
    os.makedirs(output_dir, exist_ok=True)

    fits.writeto(os.path.join(output_dir, "flat_A_map.fits"), A_map, overwrite=True)
    fits.writeto(os.path.join(output_dir, "flat_B_map.fits"), B_map, overwrite=True)
    fits.writeto(os.path.join(output_dir, "flat_C_map.fits"), C_map, overwrite=True)


def run_flat_model_fitting(master_dir: str, model_dir: str, T_ref=0.0, exp_ref=0.0):
    """
    Orchestrates the loading of master flats and fitting of pixelwise linear model.
    """
    print("\n[Step 6] Fitting parametric flat model...")
    master_map = load_master_flats(master_dir)
    A_map, B_map, C_map = fit_pixelwise_linear_flat_model(master_map, T_ref, exp_ref)
    save_flat_model(A_map, B_map, C_map, model_dir)
    print(f"[Step 6] Model saved in '{model_dir}'\n")
