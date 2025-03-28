# Author: Elías Gabriel Ferrer Jorge

"""
Step 0: Manage and create the output directories needed for subsequent steps in the pipeline.
"""

import os


def create_directories(output_base_dir: str) -> dict:
    """
    Creates the necessary subdirectories within the specified output base directory.

    :param output_base_dir: The base directory where all pipeline outputs will be stored.
    :type output_base_dir: str
    :return: A dictionary containing paths to the created subdirectories:
                 - 'dark_corrected_dir': Directory for bias-corrected dark frames.
                 - 'dark_masks_dir': Directory for generated dark masks by temperature.
                 - 'model_dir': Directory for model-related outputs (A_map, B_map, etc.).
    :rtype: dict
    """
    os.makedirs(output_base_dir, exist_ok=True)

    dark_corrected_dir = os.path.join(output_base_dir, "bias_corrected_darks")
    dark_masks_dir = os.path.join(output_base_dir, "dark_masks_by_temp")
    model_dir = os.path.join(output_base_dir, "dark_model")

    os.makedirs(dark_corrected_dir, exist_ok=True)
    os.makedirs(dark_masks_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    dirs = {
        'dark_corrected_dir': dark_corrected_dir,
        'dark_masks_dir': dark_masks_dir,
        'model_dir': model_dir
    }
    return dirs
