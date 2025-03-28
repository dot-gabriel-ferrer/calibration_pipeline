# Author: Elías Gabriel Ferrer Jorge

"""
Step 0: Manage and create the output directories needed for subsequent steps in the flat pipeline.

Directories:
    - flat_raw_reduced/: Flats after subtracting the best matching dark.
    - flat_normalized/: Flats normalized by user-selected method (max or mean).
    - flat_vignetting_corrected/: Flats after optical vignetting correction (e.g., lensfun).
    - master_flat/: Final master flats grouped by temperature & exposure time.
    - flat_model/: Outputs of any pixel-wise or parametric modeling steps.
"""

import os
from tqdm import tqdm

def create_directories(output_base_dir: str) -> dict:
    """
    Creates the necessary subdirectories within the specified output base directory for flat modeling.

    :param output_base_dir: The base directory where all pipeline outputs will be stored.
    :return: A dictionary containing paths to the created subdirectories:
                - 'flat_raw_reduced_dir'
                - 'flat_normalized_dir'
                - 'flat_vignetting_dir'
                - 'master_flat_dir'
                - 'flat_model_dir'
    """
    print("\n[Step 0: Flat Pipeline] Creating output directory structure...")
    subdirs = {
        'flat_raw_reduced_dir': os.path.join(output_base_dir, "flat_raw_reduced"),
        'flat_normalized_dir': os.path.join(output_base_dir, "flat_normalized"),
        'flat_vignetting_dir': os.path.join(output_base_dir, "flat_vignetting_corrected"),
        'master_flat_dir': os.path.join(output_base_dir, "master_flat"),
        'flat_model_dir': os.path.join(output_base_dir, "flat_model")
    }

    for desc, path in tqdm(subdirs.items(), desc="Creating directories", ncols=80):
        os.makedirs(path, exist_ok=True)

    return subdirs
