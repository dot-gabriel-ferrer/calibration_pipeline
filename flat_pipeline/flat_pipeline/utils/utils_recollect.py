# utils_recollect.py
# Author: Elías Gabriel Ferrer Jorge

"""
Utility to scan a directory and rebuild a list of flat entries 
with metadata required by the pipeline (FILTER, temperature, exposure, etc).
Used after step4 when we no longer have the original observation manager.
"""

import os
from astropy.io import fits
from tqdm import tqdm

def build_entries_from_dir(directory: str) -> list:
    """
    Scans the given directory for FITS files and extracts relevant metadata
    from each header. Builds a list of entries compatible with pipeline inputs.

    :param directory: Directory containing FITS files
    :return: List of dictionaries with keys: original_path, FILTER, temperature, exposure, etc.
    """
    entries = []
    fits_files = [f for f in os.listdir(directory) if f.lower().endswith('.fits')]

    for fname in tqdm(fits_files, desc=f"Scanning {os.path.basename(directory)}", ncols=80):
        path = os.path.join(directory, fname)
        try:
            with fits.open(path) as hdul:
                header = hdul[0].header

            entry = {
                'original_path': path,
                'filename': fname,
                'FILTER': header.get('F_SAMP', 'UNKNOWN'),
                'temperature': header.get('T_AVG', header.get('TEMP', -999)),
                'exposure': header.get('E_AVG', header.get('EXPTIME', -1)),
            }
            entries.append(entry)

        except Exception as e:
            print(f"[Warning] Could not read {fname}: {e}")

    return entries
