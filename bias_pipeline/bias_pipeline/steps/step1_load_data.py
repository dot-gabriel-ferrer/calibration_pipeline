# Author: Elías Gabriel Ferrer Jorge

"""
Step 1: Load and filter bias observations from a dataset using the ObservationManager.
This step prepares a list of valid bias files (short exposure calibration frames)
to be used in model construction.
"""

from steps.observation_manager.observation_manager import ObservationManager
from tqdm import tqdm


def load_bias_files(base_path: str):
    """
    Loads and filters bias FITS files from the specified base directory.

    The ObservationManager is used to scan and organize all available FITS files,
    then filters out the short-exposure calibration frames (bias frames).

    :param base_path: Path to the directory containing raw FITS files.
    :type base_path: str
    :return: Tuple of (manager, bias_files):
             - manager: The ObservationManager instance used for organizing.
             - bias_files: List of filtered bias file entries (dicts).
    :rtype: tuple[ObservationManager, list[dict]]
    """
    print("\n[Step 1] Loading and filtering bias observations...")
    
    # Initialize manager and organize files
    manager = ObservationManager(base_path=base_path)
    manager.load_and_organize()

    # Filter short-exposure calibration files classified as 'bias'
    bias_files = manager.filter_files(
        category='CALIBRATION',
        subcat='bias',
        exp_max=0.1,
        ext_filter='fits'
    )

    # Show how many valid bias files were found
    print(f"[Step 1] Found {len(bias_files)} bias files.\n")
    
    return manager, bias_files
