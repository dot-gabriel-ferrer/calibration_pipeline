# Author: Elías Gabriel Ferrer Jorge

"""
Main Pipeline: Bias Modeling and Evaluation

This script defines the high-level pipeline to model and evaluate temperature-dependent bias
signal behavior in CCD/CMOS sensors. It orchestrates the following steps:

1. Creates an organized output directory structure.
2. Loads and filters bias calibration frames from raw FITS observations.
3. Groups those frames by temperature and generates master bias images.
4. Fits a pixel-wise linear model (bias = a + b*T) per pixel.
5. Optionally generates synthetic bias frames using the model.
6. Evaluates model accuracy against the real master bias frames (MAE/MAPE).

Optional arguments allow hot pixel masking, controlling whether to generate synthetic bias
sets or save FITS outputs during evaluation.

This module can be used from a wrapper or CLI launcher.
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
    Executes the complete temperature-dependent bias modeling pipeline.

    Parameters
    ----------
    observations_dir : str
        Path to directory containing raw FITS calibration frames (bias).

    output_dir : str
        Path where the pipeline outputs will be stored, including models and diagnostics.

    hot_pixel_mask_path : str, optional
        Path to a 2D boolean FITS file indicating hot pixels to exclude from model fit.
        Hot pixels are masked using NaNs in the a/b coefficient maps.

    generate_set : bool, default=True
        If True, generate synthetic bias frames for each available master bias temperature.
        These are saved to disk using the model parameters.

    eval_save_fits : bool, default=False
        If True, save intermediate FITS files during model evaluation: synthetic, MAE, MAPE.

    Returns
    -------
    None
    """
    # Step 0: Create directory structure
    dirs = create_directories(output_dir)

    # Step 1: Load bias calibration frames
    manager, bias_entries = load_bias_files(observations_dir)

    # Step 2: Group and generate master biases per temperature
    master_bias_dict = generate_master_bias_by_temp(
        bias_entries, output_dir=dirs['bias_data_dir']
    )

    # Step 3: Fit linear bias model
    hot_pixel_mask = None
    if hot_pixel_mask_path and os.path.exists(hot_pixel_mask_path):
        hot_pixel_mask = fits.getdata(hot_pixel_mask_path).astype(bool)

    a_map, b_map = fit_bias_model(master_bias_dict, hot_pixel_mask=hot_pixel_mask)
    save_bias_model(a_map, b_map, output_dir=dirs['model_dir'])

    # Step 4: Optionally generate synthetic bias frames
    if generate_set:
        generate_multiple_synthetic_biases(
            a_map, b_map, list(master_bias_dict.keys()), output_dir=dirs['bias_masks_dir']
        )

    # Step 5: Evaluate model performance (plots + stats)
    evaluate_model(
        a_map, b_map, master_bias_dict,
        output_dir=os.path.join(output_dir, "evaluation"),
        save_fits=eval_save_fits
    )
