# Author: Elías Gabriel Ferrer Jorge

"""
Step 4: Generate synthetic bias frames based on the fitted model and desired temperature(s).

This step uses the pixel-wise linear model (bias = a + b*T) to generate synthetic
bias frames for given temperature values.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm


def generate_synthetic_bias(a_map: np.ndarray, b_map: np.ndarray, temperature: float) -> np.ndarray:
    """
    Generates a synthetic bias frame for a given temperature using the model maps.

    :param a_map: 2D array of intercepts (bias offset).
    :param b_map: 2D array of temperature coefficients.
    :param temperature: Temperature at which to generate the synthetic bias.
    :return: 2D array representing the synthetic bias frame.
    """
    return a_map + b_map * temperature


def generate_multiple_synthetic_biases(a_map: np.ndarray, b_map: np.ndarray,
                                       temperatures: list, output_dir: str):
    """
    Generates and saves synthetic bias frames for a list of input temperatures.

    :param a_map: 2D array of model intercepts.
    :param b_map: 2D array of temperature coefficients.
    :param temperatures: List of temperature values to generate synthetic bias frames for.
    :param output_dir: Directory where synthetic FITS files will be saved.
    """
    print("\n[Step 4] Generating synthetic bias frames...")
    os.makedirs(output_dir, exist_ok=True)

    for temp in tqdm(sorted(temperatures), desc="Generating biases", ncols=80):
        synthetic_bias = generate_synthetic_bias(a_map, b_map, temp)
        output_path = os.path.join(output_dir, f"synthetic_bias_{temp:.1f}C.fits")
        fits.writeto(output_path, synthetic_bias.astype(np.float32), overwrite=True)

    print(f"[Step 4] Saved {len(temperatures)} synthetic bias frames to '{output_dir}'\n")
