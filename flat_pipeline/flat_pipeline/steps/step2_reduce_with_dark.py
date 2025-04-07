# Author: Elías Gabriel Ferrer Jorge

"""
Step 2: Subtract the most suitable dark frame from each flat.

This step attempts to find the closest dark in temperature and exposure
and subtracts it from the flat, producing a 'raw reduced' flat.
No separate bias correction is applied, since we assume the dark includes bias.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from ..utils.utils_scaling import load_fits_scaled_12bit

def group_flats_by_filter(flat_files, filter_keyword='FILTER'):
    """
    Groups the input flat_files by the value of the specified filter keyword.
    You may also want to further group by temperature or exposure if needed.

    :param flat_files: List of dict describing each flat file (metadata + path).
    :param filter_keyword: FITS header keyword for filter.
    :return: dict mapping filter_value -> list of flat entries
    """
    grouped = {}
    for entry in flat_files:
        fvalue = entry.get(filter_keyword, 'UNKNOWN')
        if fvalue not in grouped:
            grouped[fvalue] = []
        grouped[fvalue].append(entry)
    return grouped


def find_best_matching_dark(dark_list, target_temp, target_exptime,
                            max_temp_diff=2.0, max_exp_diff=5.0):
    """
    Finds the dark frame whose temperature and exposure are closest to (target_temp, target_exptime).
    This is a simplistic approach: you can refine it with weighting or more complex logic.

    :param dark_list: List of dark dict entries with 'temperature' and 'exposure'.
    :param target_temp: Desired temperature.
    :param target_exptime: Desired exposure time.
    :param max_temp_diff: Acceptable difference in temperature (°C).
    :param max_exp_diff: Acceptable difference in exposure (seconds).
    :return: The dict entry of the best matching dark, or None if none found.
    """
    best_dark = None
    best_score = float('inf')
    for d in dark_list:
        dtemp = d.get('temperature', 9999)
        dexp = d.get('exposure', 9999)
        score = abs(dtemp - target_temp) + abs(dexp - target_exptime)
        if score < best_score:
            best_score = score
            best_dark = d

    # Optionally filter out if difference is too large
    if best_dark is not None:
        dtemp = best_dark['temperature']
        dexp = best_dark['exposure']
        if abs(dtemp - target_temp) > max_temp_diff or abs(dexp - target_exptime) > max_exp_diff:
            return None

    return best_dark


def subtract_dark_from_flat(flat_entry, dark_entry, output_path):
    """
    Loads the flat and dark from disk, subtracts dark from flat, and writes result to output_path.
    Also updates the FITS header to document the operation.
    """
    # Load the data from 16-bit to 12-bit scale
    flat_data = load_fits_scaled_12bit(flat_entry['original_path'])
    dark_data = load_fits_scaled_12bit(dark_entry['original_path'])

    # Subtract (flat_data - dark_data)
    reduced_data = flat_data - dark_data

    # Build new header from flat's original header plus metadata
    with fits.open(flat_entry['original_path']) as hdul_flat:
        header = hdul_flat[0].header.copy()

    # Document the dark used
    header['HIERARCH FLAT_DARK'] = os.path.basename(dark_entry['original_path'])
    header['HIERARCH FLAT_DARK_TEMP'] = (dark_entry['temperature'], "Temperature of chosen dark")
    header['HIERARCH FLAT_DARK_EXPT'] = (dark_entry['exposure'], "Exposure time of chosen dark")
    header['HIERARCH FLAT_DARK_DIFF'] = (
        f"Tdiff={abs(dark_entry['temperature']-flat_entry['temperature']):.1f}, " +
        f"Ediff={abs(dark_entry['exposure']-flat_entry['exposure']):.1f}",
        "Differences with target flat"
    )
    header['HIERARCH FLAT_PIPE_STEP'] = ("raw_reduced", "Flat pipeline step2 result")

    # Save to disk
    hdu = fits.PrimaryHDU(data=reduced_data.astype(np.float32), header=header)
    hdul_out = fits.HDUList([hdu])
    hdul_out.writeto(output_path, overwrite=True)


def reduce_flats_with_darks(flat_files, dark_files, output_dir,
                            filter_keyword='FILTER', max_temp_diff=5.0, max_exp_diff=10.0):
    """
    For each flat, finds the best matching dark (closest T and exposure),
    subtracts it, and saves the result in 'output_dir'.

    :param flat_files: list of dict describing each flat.
    :param dark_files: list of dict describing each dark.
    :param output_dir: where to store the subtracted flat (FITS).
    :param filter_keyword: how to group flats, if grouping is needed.
    :param max_temp_diff: max allowed temperature difference to accept the dark.
    :param max_exp_diff: max allowed exposure difference.
    """
    os.makedirs(output_dir, exist_ok=True)

    # If needed, group flats by filter:
    grouped = group_flats_by_filter(flat_files, filter_keyword=filter_keyword)

    # For each filter group
    for fval, entries in grouped.items():
        print(f"\n[Step 2] Processing filter = {fval}, total flats = {len(entries)}")
        for flat_entry in tqdm(entries, desc="Subtracting dark", ncols=80):
            # find best matching dark
            temp = flat_entry['temperature']
            exp_time = flat_entry['exposure']
            best_dark = find_best_matching_dark(dark_files, temp, exp_time,
                                                max_temp_diff=max_temp_diff,
                                                max_exp_diff=max_exp_diff)
            if best_dark is None:
                print(f" No suitable dark found for: {flat_entry['original_path']}")
                continue

            # build output filename
            
            file_name = os.path.basename(flat_entry['original_path']).replace(".fits", "_FLAT_REDUCED_WITH_DARK.fits")
            out_path = os.path.join(output_dir, file_name)
        

            subtract_dark_from_flat(flat_entry, best_dark, out_path)
