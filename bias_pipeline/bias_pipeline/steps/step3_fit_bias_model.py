# Author: Elías Gabriel Ferrer Jorge

"""
Step 3: Fit a temperature-dependent pixel-wise model for the bias signal.

This step performs a pixel-wise linear regression of the form:
    bias(T) = a + b*T
for each pixel, based on the master bias frames obtained at different temperatures.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm


def fit_bias_model(master_bias_dict: dict, hot_pixel_mask: np.ndarray = None) -> tuple:
    """
    Fits a linear model per pixel for bias temperature dependence: bias = a + b*T

    :param master_bias_dict: Dictionary mapping temperature (float) to 2D master bias array.
    :param hot_pixel_mask: Optional boolean 2D mask to exclude hot pixels from the model (True = hot).
    :return: Tuple (a_map, b_map) with model parameters per pixel.
    """
    print("\n[Step 3] Fitting linear bias model per pixel...")

    temperatures = sorted(master_bias_dict.keys())
    stack = np.stack([master_bias_dict[t] for t in temperatures], axis=0)  # shape: (n_temp, H, W)
    temps = np.array(temperatures)

    H, W = stack.shape[1:]
    X = np.vstack([np.ones_like(temps), temps]).T  # shape: (n_temp, 2)

    # Flatten the image stack for linear regression
    Y = stack.reshape(len(temps), -1)  # shape: (n_temp, H*W)

    # Perform least squares fit: beta = (X^T X)^-1 X^T Y
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # shape: (2, H*W)

    a_map = beta[0].reshape(H, W)  # Intercept map
    b_map = beta[1].reshape(H, W)  # Temperature coefficient map

    # Mask hot pixels if provided
    if hot_pixel_mask is not None:
        a_map = np.where(hot_pixel_mask, np.nan, a_map)
        b_map = np.where(hot_pixel_mask, np.nan, b_map)

    print(f"[Step 3] Model fitted for {H}x{W} pixels across {len(temps)} temperatures.\n")
    return a_map, b_map


def save_bias_model(a_map: np.ndarray, b_map: np.ndarray, output_dir: str):
    """
    Saves the bias model parameters to FITS files: bias_a_map.fits and bias_b_map.fits.

    :param a_map: 2D array of model intercepts (bias offset per pixel).
    :param b_map: 2D array of temperature coefficients (slope per pixel).
    :param output_dir: Directory where the model FITS files will be saved.
    """
    print("[Step 3] Saving bias model to disk...")
    os.makedirs(output_dir, exist_ok=True)

    fits.writeto(os.path.join(output_dir, 'bias_a_map.fits'), a_map.astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(output_dir, 'bias_b_map.fits'), b_map.astype(np.float32), overwrite=True)

    print(f"[Step 3] Saved model maps to '{output_dir}'\n")
