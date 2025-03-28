# Author: Elías Gabriel Ferrer Jorge

"""
Step 4: Correct vignetting effects on the normalized flat frames.

This step uses an external library (e.g., lensfun) or any custom method to
compensate for optical vignetting, thus providing more accurate flat frames.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

##############################
# PSEUDOCODE FOR LENSFUN
# import lensfunpy
##############################

def correct_vignetting(data: np.ndarray, lens_params: dict = None) -> np.ndarray:
    """
    Applies a vignetting correction to the given flat image data.
    This is a placeholder function. In a real scenario, you'd use lensfun or
    a custom approach to remove optical vignetting.

    :param data: 2D numpy array representing the normalized flat.
    :param lens_params: Dictionary with lens parameters, e.g. focal length, f-stop, etc.
    :return: The corrected flat as a 2D numpy array.
    """
    # Pseudocode approach:
    # 1) lens = lensfunpy.Lens(...)
    # 2) cam = lensfunpy.Camera(...)
    # 3) Modify data using lens.apply_vignetting(...)
    #    or some custom formula that depends on the distance from center
    # For now, we'll just do a dummy operation that doesn't actually do anything:
    corrected_data = data.copy()
    # Example dummy: corrected_data *= 1.0 / (1.0 + 0.01 * r^2) or something similar
    return corrected_data


def correct_vignetting_in_dir(input_dir: str, output_dir: str, 
                              lens_params: dict = None, suffix="_vigncorr.fits"):
    """
    Iterates over all normalized flats in 'input_dir', corrects vignetting,
    and saves them to 'output_dir'.

    :param input_dir: Directory with normalized flats (step3 output).
    :param output_dir: Directory to save vignetting-corrected flats.
    :param lens_params: Dictionary with lensfun or any custom vignetting config.
    :param suffix: Suffix for naming the corrected flat files.
    """
    os.makedirs(output_dir, exist_ok=True)
    fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".fits")]
    
    print(f"\n[Step 4] Correcting vignetting on {len(fits_files)} flat files.")
    for fname in tqdm(fits_files, desc="Vignetting Correction", ncols=80):
        path_in = os.path.join(input_dir, fname)
        path_out = os.path.join(output_dir, fname.replace(".fits", suffix))

        # Load the flat
        with fits.open(path_in) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()

        # Apply correction
        corrected = correct_vignetting(data, lens_params=lens_params)

        # Update header
        header['HIERARCH FLAT_VIGN'] = ("lensfun" if lens_params else "custom",
                                        "Vignetting correction method or library")
        header['HIERARCH FLAT_PIPE_STEP'] = ("vignetting_corrected", "Flat pipeline step4 result")

        # Save
        hdu = fits.PrimaryHDU(data=corrected, header=header)
        hdul_out = fits.HDUList([hdu])
        hdul_out.writeto(path_out, overwrite=True)

    print(f"[Step 4] Vignetting-corrected flats saved to '{output_dir}'\n")
