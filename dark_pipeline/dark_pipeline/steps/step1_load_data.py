# Author: Elías Gabriel Ferrer Jorge

"""
Step 1: Load observations using the ObservationManager and filter files 
for dark frames (long exposures) and bias frames (short exposures).
"""

import os
from .observation_manager.observation_manager import ObservationManager
from collections import defaultdict

def load_observations(basepath: str):
    """
    Loads and organizes observation files from the specified base directory,
    then filters them to obtain two lists: long dark frames and short dark (bias) frames.

    :param basepath: The path where the observation FITS files are located.
    :type basepath: str
    :return: A tuple (manager, long_darks, short_darks):
                - manager: the ObservationManager instance used for file organization
                - long_darks: list of dictionaries describing each long dark file
                - short_darks: list of dictionaries describing each short dark (bias) file
    :rtype: tuple
    """
    manager = ObservationManager(base_path=basepath)
    manager.load_and_organize()

    # Filter for calibration dark frames (long exposures)
    long_darks = manager.filter_files(
        category='CALIBRATION',
        subcat='darks',
        exp_min=0.1,  # Example threshold
        ext_filter='fits'
    )

    # Filter for bias frames (very short exposures)
    short_darks = manager.filter_files(
        category='CALIBRATION',
        subcat='bias',
        exp_max=0.1,  # Example threshold
        ext_filter='fits'
    )

    # Group long darks by exposure time
    grouped_darks = defaultdict(list)
    for entry in long_darks:
        exp_time = round(entry['exposure'], 2)  # Optional: round to avoid float noise
        grouped_darks[exp_time].append(entry)

    return manager, grouped_darks, short_darks
