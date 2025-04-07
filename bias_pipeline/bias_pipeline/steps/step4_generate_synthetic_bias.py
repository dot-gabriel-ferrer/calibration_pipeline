# Author: Elías Gabriel Ferrer Jorge

"""
Step 4: Generate Synthetic Bias Frames from Fitted Model

This module uses the fitted pixel-wise bias model parameters (intercept and slope maps)
to generate synthetic bias images for any given temperature.

The model assumes:
    bias(T) = a + b*T

These synthetic frames can be used for calibration when no matching master bias exists.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def generate_synthetic_bias(a_map: np.ndarray, b_map: np.ndarray, temperature: float) -> np.ndarray:
    """
    Compute a synthetic bias image for a specific temperature.

    Parameters:
    ------------
    a_map : np.ndarray
        Pixel-wise bias offset at 0°C (intercept map).

    b_map : np.ndarray
        Pixel-wise temperature sensitivity (slope map).

    temperature : float
        Temperature at which to evaluate the model.

    Returns:
    --------
    np.ndarray
        2D synthetic bias image evaluated at the given temperature.
    """
    return a_map + b_map * temperature

def generate_multiple_synthetic_biases(a_map: np.ndarray, b_map: np.ndarray,
                                       temperatures: list, output_dir: str):
    """
    Generate and save synthetic bias images for a list of temperatures.

    Parameters:
    ------------
    a_map : np.ndarray
        Model intercepts (bias level at T=0).

    b_map : np.ndarray
        Model slopes (rate of bias change with T).

    temperatures : list
        List of float values for which synthetic bias frames will be generated.

    output_dir : str
        Directory where the synthetic FITS files will be stored.
    """
    print("\n[Step 4] Generating synthetic bias frames...")
    os.makedirs(output_dir, exist_ok=True)

    for temp in tqdm(sorted(temperatures), desc="Generating biases", ncols=80):
        synthetic_bias = generate_synthetic_bias(a_map, b_map, temp)
        output_path = os.path.join(output_dir, f"synthetic_bias_{temp:.1f}C.fits")
        fits.writeto(output_path, synthetic_bias.astype(np.float32), overwrite=True)

    print(f"[Step 4] Saved {len(temperatures)} synthetic bias frames to '{output_dir}'\n")