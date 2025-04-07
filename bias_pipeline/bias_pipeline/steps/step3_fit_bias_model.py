# Author: Elías Gabriel Ferrer Jorge

"""
Step 3: Fit a Temperature-Dependent Pixel-wise Bias Model

This module fits a pixel-by-pixel linear model to describe how the bias level changes
as a function of sensor temperature. For each pixel, the model is:

    bias(T) = a + b * T

Where:
- a = offset (bias at 0°C)
- b = temperature slope (sensitivity of pixel to temperature)

This is typically used to generate synthetic bias frames at arbitrary temperatures.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def fit_bias_model(master_bias_dict: dict, hot_pixel_mask: np.ndarray = None) -> tuple:
    """
    Fit a per-pixel linear regression model across temperature.

    Parameters:
    ------------
    master_bias_dict : dict
        Dictionary mapping temperature (float) to 2D master bias frame.

    hot_pixel_mask : np.ndarray, optional
        Boolean array (H, W). If provided, pixels marked True are excluded (set to NaN).

    Returns:
    ---------
    tuple[np.ndarray, np.ndarray]
        a_map : 2D array of intercepts per pixel.
        b_map : 2D array of slopes (temperature coefficients) per pixel.
    """
    print("\n[Step 3] Fitting linear bias model per pixel...")

    temperatures = sorted(master_bias_dict.keys())
    stack = np.stack([master_bias_dict[t] for t in temperatures], axis=0)  # shape: (N, H, W)
    temps = np.array(temperatures)

    H, W = stack.shape[1:]
    X = np.vstack([np.ones_like(temps), temps]).T  # shape: (N, 2)
    Y = stack.reshape(len(temps), -1)              # shape: (N, H*W)

    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # shape: (2, H*W)

    a_map = beta[0].reshape(H, W)  # Intercepts
    b_map = beta[1].reshape(H, W)  # Slopes

    if hot_pixel_mask is not None:
        a_map = np.where(hot_pixel_mask, np.nan, a_map)
        b_map = np.where(hot_pixel_mask, np.nan, b_map)

    print(f"[Step 3] Model fitted for {H}x{W} pixels across {len(temps)} temperatures.\n")
    return a_map, b_map

def save_bias_model(a_map: np.ndarray, b_map: np.ndarray, output_dir: str):
    """
    Save pixel-wise bias model coefficients to FITS files.

    Parameters:
    ------------
    a_map : np.ndarray
        Intercept map (bias at T=0°C).

    b_map : np.ndarray
        Slope map (change of bias with temperature).

    output_dir : str
        Directory to save output FITS files.
    """
    print("[Step 3] Saving bias model to disk...")
    os.makedirs(output_dir, exist_ok=True)

    fits.writeto(os.path.join(output_dir, 'bias_a_map.fits'), a_map.astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(output_dir, 'bias_b_map.fits'), b_map.astype(np.float32), overwrite=True)

    print(f"[Step 3] Saved model maps to '{output_dir}'\n")
