# Author: Elías Gabriel Ferrer Jorge

"""
Step 5: Generate final Master Flats from the vignetting-corrected flats.

This step groups flats by filter, temperature, exposure time (or other criteria),
and combines each group into a single 'master flat' using a chosen method 
(e.g., median or mean). The resulting FITS file includes detailed metadata 
documenting the input files, statistics, and pipeline steps.
"""

import os
import numpy as np
from astropy.io import fits
from collections import defaultdict
from tqdm import tqdm


def group_flats_for_master(flat_entries, grouping=('FILTER', 'temperature', 'exposure')):
    """
    Groups the list of flat file dictionaries by the specified grouping keys.
    By default, groups by (FILTER, temperature, exposure).

    :param flat_entries: list of dict, each describing a flat (metadata + path).
    :param grouping: tuple/list of keys to group by, e.g. ('FILTER', 'temperature', 'exposure').
    :return: A dict with keys = (filter, temp, exp) and values = list of entries in that group.
    """
    groups = defaultdict(list)
    for entry in flat_entries:
        # Build a grouping key from the desired metadata
        group_key = []
        for key in grouping:
            group_key.append(entry.get(key, "UNKNOWN"))
        group_key = tuple(group_key)
        groups[group_key].append(entry)

    return dict(groups)


def combine_master_flat(file_entries, method='median'):
    """
    Loads each flat in 'file_entries', stacks them, and combines 
    with the chosen method (median or mean).

    :param file_entries: list of dict, each must have 'original_path' 
                         pointing to the vignetting-corrected flat.
    :param method: 'median' or 'mean'
    :return: (combined_data, combined_header) 
             combined_data = 2D np.array 
             combined_header = aggregated fits.Header
    """
    data_stack = []
    # We'll also collect some info for the header
    used_paths = []
    temps = []
    exps = []
    filters_used = []

    for entry in file_entries:
        path = entry['original_path']
        used_paths.append(os.path.basename(path))
        temps.append(entry.get('temperature', -999))
        exps.append(entry.get('exposure', -1))
        filters_used.append(str(entry.get('FILTER', 'UNKNOWN')))

        with fits.open(path) as hdul:
            data_stack.append(hdul[0].data.astype(np.float32))

    data_stack = np.stack(data_stack, axis=0)

    if method == 'median':
        combined_data = np.median(data_stack, axis=0)
    elif method == 'mean':
        combined_data = np.mean(data_stack, axis=0)
    else:
        raise ValueError("Unsupported combination method. Use 'median' or 'mean'.")

    # Build an aggregated header 
    combined_header = fits.Header()
    combined_header['FLATPIPE'] = ('master_flat', 'Pipeline step5 result')
    combined_header['N_FILES'] = (len(file_entries), 'Number of combined flats')
    combined_header['COMBMETH'] = (method.upper(), 'Combination method used')

    # Document min, max, mean of the final master
    combined_header['DATAMIN'] = float(np.nanmin(combined_data))
    combined_header['DATAMAX'] = float(np.nanmax(combined_data))
    combined_header['DATAMEAN'] = float(np.nanmean(combined_data))

    # Just an example: store the first and last file name used
    combined_header['SRC1'] = used_paths[0]
    combined_header['SRCLAST'] = used_paths[-1]

    # Document typical temperature, exposure, filter (just as an example)
    combined_header['T_AVG'] = float(np.mean(temps))
    combined_header['E_AVG'] = float(np.mean(exps))
    combined_header['F_SAMP'] = filters_used[0]

    # Optionally store a whole list of paths or other stats
    # (You could do something like combined_header['SRCLIST'] = str(used_paths) if not too long)

    return combined_data.astype(np.float32), combined_header


def generate_master_flats(flat_entries, output_dir, 
                          grouping=('FILTER', 'temperature', 'exposure'),
                          method='median'):
    """
    Main function to generate master flats from a list of vignetting-corrected flat entries.

    :param flat_entries: list of dict, each must have 'original_path' 
                         to the vignetting-corrected flat, plus keys for grouping 
                         (like temperature, exposure, FILTER).
    :param output_dir: Where to save the master flats.
    :param grouping: Keys to group by (default: filter, temperature, exposure).
    :param method: 'median' or 'mean'
    """
    print("\n[Step 5] Generating Master Flats...")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Group the entries
    groups = group_flats_for_master(flat_entries, grouping=grouping)

    # 2) Combine each group
    for group_key, entries in tqdm(groups.items(), desc="Combining flats", ncols=80):
        combined_data, combined_header = combine_master_flat(entries, method=method)

        # Build an output filename 
        # e.g. group_key = (filter, temp, exp)
        # might produce "master_flat_FILTER_tempX_expY.fits"
        # adapt naming as desired
        fval = str(group_key[0]).replace(" ", "")
        tval = float(group_key[1])
        eval_ = float(group_key[2])
        out_name = f"master_flat_{fval}_T{tval:.1f}_E{eval_:.1f}.fits"

        out_path = os.path.join(output_dir, out_name)
        
        # Save result
        hdu = fits.PrimaryHDU(data=combined_data, header=combined_header)
        hdul_out = fits.HDUList([hdu])
        hdul_out.writeto(out_path, overwrite=True)

    print(f"[Step 5] Master flats saved to '{output_dir}'\n")
