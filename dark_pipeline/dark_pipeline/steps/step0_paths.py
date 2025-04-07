# Author: Elías Gabriel Ferrer Jorge

"""
Step 0: Manage and create the output directories needed for subsequent steps in the dark current modeling pipeline.

This script initializes the necessary folder structure used throughout the pipeline.
Its purpose is to guarantee that all future steps have well-defined, writable destinations
for their outputs. This ensures modularity, organization, and reproducibility of results.
"""

import os


def create_directories(output_base_dir: str) -> dict:
    """
    Creates the required subdirectory structure within a given output base directory.

    The structure separates outputs from different processing stages:
    - Bias-corrected dark frames
    - Dark current masks (generated per temperature)
    - Fitted dark model outputs

    Parameters
    ----------
    output_base_dir : str
        The base directory under which all pipeline results will be stored.

    Returns
    -------
    dict
        A dictionary containing absolute paths to each created subdirectory:
            - 'dark_corrected_dir': path to store dark frames corrected for bias.
            - 'dark_masks_dir': path to store modeled dark masks (per temperature).
            - 'model_dir': path to store the fitted parameter maps of the dark model.
    """

    # Ensure the root output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Define paths to subdirectories for each processing stage
    dark_corrected_dir = os.path.join(output_base_dir, "bias_corrected_darks")
    dark_masks_dir = os.path.join(output_base_dir, "dark_masks_by_temp")
    model_dir = os.path.join(output_base_dir, "dark_model")

    # Create each subdirectory if it doesn't already exist
    os.makedirs(dark_corrected_dir, exist_ok=True)
    os.makedirs(dark_masks_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Return dictionary with resolved paths to the subdirectories
    dirs = {
        'dark_corrected_dir': dark_corrected_dir,
        'dark_masks_dir': dark_masks_dir,
        'model_dir': model_dir
    }
    return dirs
