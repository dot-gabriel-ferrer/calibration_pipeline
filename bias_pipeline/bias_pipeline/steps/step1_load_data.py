# Author: Elías Gabriel Ferrer Jorge

"""
Step 1: Load and Filter Bias Observations

This module uses the ObservationManager to recursively scan a directory
for bias calibration frames. It loads metadata, classifies each FITS file,
and selects only short-exposure calibration frames marked as 'bias'.

The filtered list of valid bias files is then passed downstream for modeling.
"""

from steps.observation_manager.observation_manager import ObservationManager
from tqdm import tqdm

def load_bias_files(base_path: str):
    """
    Load and filter valid bias calibration files from a raw FITS dataset.

    The filtering is handled by the ObservationManager and targets files labeled:
    - Category: CALIBRATION
    - Subcategory: bias
    - Exposure time: <= 0.1 seconds
    - Extension: .fits only

    Parameters:
    ------------
    base_path : str
        Path to the directory containing the raw FITS calibration files.

    Returns:
    --------
    tuple[ObservationManager, list[dict]]
        manager     : ObservationManager instance with all file metadata loaded.
        bias_files  : List of filtered FITS entries (dicts) representing usable bias frames.
    """
    print("\n[Step 1] Loading and filtering bias observations...")

    # Load and classify all FITS files
    manager = ObservationManager(base_path=base_path)
    manager.load_and_organize()

    # Select short-exposure bias frames
    bias_files = manager.filter_files(
        category='CALIBRATION',
        subcat='bias',
        exp_max=0.1,
        ext_filter='fits'
    )

    print(f"[Step 1] Found {len(bias_files)} bias files.\n")
    return manager, bias_files