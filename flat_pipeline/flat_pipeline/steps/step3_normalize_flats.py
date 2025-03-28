# Author: Elías Gabriel Ferrer Jorge

"""
Step 3: Normalize 'raw reduced' flat frames by max or mean.

This step is crucial to transform the flats into a standard scale, either by
dividing by the maximum pixel value or by the mean pixel value. The choice is
user-defined. The normalized flats are saved in flat_normalized/ with updated headers.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def normalize_flat(input_path: str, output_path: str, method: str = "max"):
    """
    Normalize a single flat frame by its max or mean pixel value.

    :param input_path: Path to the raw reduced flat FITS file.
    :param output_path: Where to save the normalized flat.
    :param method: 'max' or 'mean' for normalization.
    """
    with fits.open(input_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()

    if method.lower() == "max":
        norm_factor = float(np.nanmax(data))
        norm_str = "MAX"
    elif method.lower() == "mean":
        norm_factor = float(np.nanmean(data))
        norm_str = "MEAN"
    else:
        raise ValueError("Normalization method must be either 'max' or 'mean'")

    # Avoid division by zero
    if norm_factor == 0.0:
        # In case there's an unexpected flat that's fully zero
        norm_factor = 1.0

    normalized_data = data / norm_factor

    # Update header to document the normalization
    header['HIERARCH FLAT_NORM'] = (norm_str, "Method used for normalization")
    header['HIERARCH FLAT_NFAC'] = (norm_factor, "Normalization factor")
    header['HIERARCH FLAT_PIPE_STEP'] = ("normalized", "Flat pipeline step3 result")

    # Save result
    hdu = fits.PrimaryHDU(data=normalized_data, header=header)
    hdul_out = fits.HDUList([hdu])
    hdul_out.writeto(output_path, overwrite=True)


def normalize_flats_in_dir(raw_reduced_dir: str, output_dir: str, 
                           method: str = "max", file_suffix="_normalized.fits"):
    """
    Iterates over all FITS files in raw_reduced_dir, normalizes them, and
    saves in output_dir with the chosen method.

    :param raw_reduced_dir: Path to the directory with raw reduced flats (from step2).
    :param output_dir: Path where normalized flats will be stored.
    :param method: Normalization strategy: 'max' or 'mean'.
    :param file_suffix: Suffix added to output filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    fits_files = [f for f in os.listdir(raw_reduced_dir) if f.lower().endswith(".fits")]

    print(f"\n[Step 3] Normalizing {len(fits_files)} flat files by {method.upper()}.")
    for fname in tqdm(fits_files, desc="Normalizing flats", ncols=80):
        input_path = os.path.join(raw_reduced_dir, fname)
        out_name = fname.replace(".fits", file_suffix)
        output_path = os.path.join(output_dir, out_name)

        normalize_flat(input_path, output_path, method=method)

    print(f"[Step 3] Normalized flats saved to '{output_dir}'\n")
