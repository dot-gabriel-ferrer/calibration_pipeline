# Author: Elías Gabriel Ferrer Jorge

"""
Step 2: Generate master bias frames grouped by sensor temperature.

This step groups the bias FITS files by temperature and combines them
(e.g. using median or mean) to generate representative master bias frames.
"""

import os
import numpy as np
from collections import defaultdict
from astropy.io import fits
from tqdm import tqdm
from steps.utils.utils_scaling import load_fits_scaled_12bit

def group_by_temperature(file_list, round_decimals: int = 1):
    """
    Groups FITS files by their temperature values, rounded to the specified number of decimals.

    :param file_list: List of file dictionaries containing metadata (including 'temperature').
    :type file_list: list
    :param round_decimals: Number of decimals to round the temperature values.
    :type round_decimals: int
    :return: Dictionary with temperatures as keys and lists of files as values.
    :rtype: dict[float, list]
    """
    temp_groups = defaultdict(list)
    for entry in file_list:
        temp = round(entry['temperature'], round_decimals)
        temp_groups[temp].append(entry)
    return temp_groups


def combine_frames(file_entries, method: str = 'median') -> np.ndarray:
    """
    Loads image data from a list of FITS files and combines them using the specified method.

    :param file_entries: List of file entries, each containing 'original_path'.
    :type file_entries: list
    :param method: Combination method: 'median' or 'mean'.
    :type method: str
    :return: Combined 2D numpy array representing the master bias.
    :rtype: np.ndarray
    """
    data_stack = []
    for entry in file_entries:
        data_stack.append(load_fits_scaled_12bit(entry['original_path']))

    data_stack = np.stack(data_stack, axis=0)

    if method == 'median':
        return np.median(data_stack, axis=0)
    elif method == 'mean':
        return np.mean(data_stack, axis=0)
    else:
        raise ValueError("Unsupported combination method. Use 'median' or 'mean'.")


def generate_master_bias_by_temp(bias_entries: list, output_dir: str, method: str = 'median') -> dict:
    """
    Generates master bias frames for each temperature group and saves them to disk.

    :param bias_entries: List of dictionaries with metadata and paths for bias files.
    :param output_dir: Directory where the master bias FITS files will be saved.
    :param method: Combination method to use ('median' or 'mean').
    :return: Dictionary mapping each temperature to its master bias 2D array.
    """
    print("\n[Step 2] Generating master bias frames by temperature...")
    os.makedirs(output_dir, exist_ok=True)

    grouped = group_by_temperature(bias_entries)
    master_bias_dict = {}

    for temp in tqdm(sorted(grouped.keys()), desc="Processing temperatures", ncols=80):
        files = grouped[temp]
        master_bias = combine_frames(files, method=method)
        master_bias_dict[temp] = master_bias

        # Save to FITS
        output_path = os.path.join(output_dir, f"master_bias_{temp:.1f}C.fits")
        fits.writeto(output_path, master_bias.astype(np.float32), overwrite=True)

    print(f"[Step 2] Saved {len(master_bias_dict)} master bias frames to '{output_dir}'\n")
    return master_bias_dict
