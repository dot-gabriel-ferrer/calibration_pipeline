# Author: Elías Gabriel Ferrer Jorge

"""
Step 2: Generate Master Bias Frames Grouped by Temperature

This step groups raw bias calibration frames based on the sensor temperature at capture time.
For each temperature group, it combines multiple exposures into a representative master bias frame
using either a median or mean operation.

These master biases are saved as individual FITS files named according to their temperature group.
"""

import os
import numpy as np
from collections import defaultdict
from astropy.io import fits
from tqdm import tqdm
from .utils.utils_scaling import load_fits_scaled_12bit

def group_by_temperature(file_list, round_decimals: int = 1):
    """
    Cluster a list of FITS metadata entries by rounded temperature value.

    Parameters:
    ------------
    file_list : list
        List of observation metadata dictionaries, each with a 'temperature' key.

    round_decimals : int
        Number of decimal digits to round the temperature.

    Returns:
    --------
    dict[float, list]
        A dictionary mapping rounded temperature to corresponding bias file entries.
    """
    temp_groups = defaultdict(list)
    for entry in file_list:
        temp = round(entry['temperature'], round_decimals)
        temp_groups[temp].append(entry)
    return temp_groups

def combine_frames(file_entries, method: str = 'median') -> np.ndarray:
    """
    Load FITS files and stack them to compute a combined image using median or mean.

    Parameters:
    ------------
    file_entries : list
        List of FITS metadata entries with 'original_path' keys.

    method : str
        Statistical method to use for stacking ('median' or 'mean').

    Returns:
    --------
    np.ndarray
        2D array representing the master bias image for the group.
    """
    data_stack = [load_fits_scaled_12bit(entry['original_path']) for entry in file_entries]
    data_stack = np.stack(data_stack, axis=0)

    if method == 'median':
        return np.median(data_stack, axis=0)
    elif method == 'mean':
        return np.mean(data_stack, axis=0)
    else:
        raise ValueError("Unsupported combination method. Use 'median' or 'mean'.")

def generate_master_bias_by_temp(bias_entries: list, output_dir: str, method: str = 'median') -> dict:
    """
    Generate and save master bias frames per temperature group.

    Parameters:
    ------------
    bias_entries : list
        List of FITS metadata entries tagged as bias frames.

    output_dir : str
        Destination folder to store generated master biases.

    method : str
        Method used to combine the frames ('median' or 'mean').

    Returns:
    --------
    dict[float, np.ndarray]
        Dictionary mapping temperature to master bias numpy array.
    """
    print("\n[Step 2] Generating master bias frames by temperature...")
    os.makedirs(output_dir, exist_ok=True)

    grouped = group_by_temperature(bias_entries)
    master_bias_dict = {}

    for temp in tqdm(sorted(grouped.keys()), desc="Processing temperatures", ncols=80):
        files = grouped[temp]
        master_bias = combine_frames(files, method=method)
        master_bias_dict[temp] = master_bias

        output_path = os.path.join(output_dir, f"master_bias_{temp:.1f}C.fits")
        fits.writeto(output_path, master_bias.astype(np.float32), overwrite=True)

    print(f"[Step 2] Saved {len(master_bias_dict)} master bias frames to '{output_dir}'\n")
    return master_bias_dict
