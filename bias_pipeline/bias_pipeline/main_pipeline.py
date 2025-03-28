# Author: Elías Gabriel Ferrer Jorge

"""
Main pipeline script for generating and evaluating the temperature-dependent bias model.
"""

import os
from astropy.io import fits

# Import steps
from steps.step0_create_dirs import create_directories
from steps.step1_load_data import load_bias_files
from steps.step2_generate_master_bias_by_temp import generate_master_bias_by_temp
from steps.step3_fit_bias_model import fit_bias_model, save_bias_model
from steps.step4_generate_synthetic_bias import generate_multiple_synthetic_biases
from steps.step5_evaluate_model import evaluate_model


def run_bias_pipeline(observations_dir: str, output_dir: str,
                      hot_pixel_mask_path: str = None,
                      generate_set: bool = True,
                      eval_save_fits: bool = False):
    """
    Runs the full bias model pipeline.

    :param observations_dir: Path where the raw FITS files are stored.
    :param output_dir: Path to store all outputs (bias model, results, plots).
    :param hot_pixel_mask_path: Optional path to a hot pixel mask FITS file.
    :param generate_set: Whether to generate a full set of synthetic biases (step4).
    :param eval_save_fits: Whether to save FITS files for step5 evaluation.
    """
    # Step 0: Create directory structure
    dirs = create_directories(output_dir)

    # Step 1: Load observations
    manager, bias_entries = load_bias_files(observations_dir)

    # Step 2: Generate master bias by temperature
    master_bias_dict = generate_master_bias_by_temp(
        bias_entries, output_dir=dirs['bias_data_dir']
    )

    # Step 3: Fit model
    hot_pixel_mask = None
    if hot_pixel_mask_path and os.path.exists(hot_pixel_mask_path):
        hot_pixel_mask = fits.getdata(hot_pixel_mask_path).astype(bool)

    a_map, b_map = fit_bias_model(master_bias_dict, hot_pixel_mask=hot_pixel_mask)
    save_bias_model(a_map, b_map, output_dir=dirs['model_dir'])

    # Step 4: Generate synthetic biases (optional full set)
    if generate_set:
        generate_multiple_synthetic_biases(
            a_map, b_map, list(master_bias_dict.keys()), output_dir=dirs['bias_masks_dir']
        )

    # Step 5: Evaluate model
    evaluate_model(
        a_map, b_map, master_bias_dict,
        output_dir=os.path.join(output_dir, "evaluation"),
        save_fits=eval_save_fits
    )
