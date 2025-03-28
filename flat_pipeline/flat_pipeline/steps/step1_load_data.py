# Author: Elías Gabriel Ferrer Jorge

"""
Step 1: Load and filter flat frames from a dataset using the ObservationManager.

This step retrieves all FLAT exposures from a given base path, organizing
and storing essential metadata such as temperature, exposure time, filter, etc.
"""

import os
from tqdm import tqdm
from steps.observation_manager.observation_manager import ObservationManager

def load_flat_files(base_path: str, filter_keyword: str = "FILTER"):
    """
    Loads and filters flat FITS files from the specified base directory.

    The ObservationManager is used to scan and organize all available FITS files,
    then filter out the calibration frames classified as 'flat'.

    :param base_path: Path to the directory containing the flat FITS files.
    :type base_path: str
    :param filter_keyword: FITS header keyword that indicates the filter used (e.g. 'FILTER').
    :type filter_keyword: str
    :return: Tuple (manager, flat_files), where:
             - manager is the ObservationManager instance used for organization
             - flat_files is a list of file entries (dicts) representing each flat exposure
               Each entry typically contains keys like:
                 'original_path', 'temperature', 'exposure', filter_keyword, etc.
    :rtype: tuple[ObservationManager, list[dict]]
    """
    print("\n[Step 1: Flat Pipeline] Loading flat frames...")
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    manager = ObservationManager(base_path=base_path)
    manager.load_and_organize()

    # Filter for calibration flat frames
    # (Adjust category/subcat if needed to match your classification)
    flat_files = manager.filter_files(
        category='CALIBRATION',
        subcat='flat',
        ext_filter='fits'
    )

    print(f"[Step 1] Found {len(flat_files)} flat files in '{base_path}'.\n")
    
    # (Optional) log some details
    for entry in tqdm(flat_files, desc="Logging flat files", ncols=80):
        # Example usage: check the filter from the header
        filt = entry.get(filter_keyword, "UNKNOWN")
        temp = entry.get('temperature', "N/A")
        exp_time = entry.get('exposure', "N/A")
        # Optionally print or store more detail
        # print(f" - {entry['original_path']} | Filter={filt}, T={temp}°C, Exp={exp_time}s")

    return manager, flat_files
