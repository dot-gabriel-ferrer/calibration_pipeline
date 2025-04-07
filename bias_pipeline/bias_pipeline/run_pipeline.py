# Author: Elías Gabriel Ferrer Jorge

"""
Run pipeline script for executing the bias model pipeline end-to-end.

This command-line utility executes the complete temperature-dependent bias modeling pipeline,
which includes the following steps:

    1. Create necessary output directory structure.
    2. Load and filter valid bias FITS files from a directory.
    3. Generate master bias frames grouped by sensor temperature.
    4. Fit a linear temperature-dependent model per pixel (bias = a + b*T).
    5. Optionally generate synthetic biases using the model.
    6. Evaluate model performance (e.g., MAE, MAPE) with plots and stats.

Usage Example:
    python run_pipeline.py \
        --basepath /path/to/raw_bias_data \
        --output-dir /path/to/output_results \
        --hot-pixel-mask /path/to/hot_pixel_mask.fits \
        --generate-set \
        --save-eval-fits

Arguments:
    --basepath         Required. Directory containing raw bias FITS files.
    --output-dir       Required. Output directory where results will be stored.
    --hot-pixel-mask   Optional. Path to a FITS file with a hot pixel mask.
    --generate-set     Flag. If set, generates synthetic bias frames for each temperature.
    --save-eval-fits   Flag. If set, saves synthetic, MAE, and MAPE FITS in evaluation.
"""

import argparse
from main_pipeline import run_bias_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the bias model pipeline.")
    parser.add_argument("--basepath", type=str, required=True,
                        help="Path to the directory containing raw bias FITS files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output directory for storing model results.")
    parser.add_argument("--hot-pixel-mask", type=str, default=None,
                        help="Optional path to hot pixel mask FITS file.")
    parser.add_argument("--generate-set", action="store_true",
                        help="Generate synthetic biases for all available temperatures.")
    parser.add_argument("--save-eval-fits", action="store_true",
                        help="Save FITS files for synthetic bias, MAE, and MAPE in evaluation.")

    args = parser.parse_args()

    run_bias_pipeline(
        observations_dir=args.basepath,
        output_dir=args.output_dir,
        hot_pixel_mask_path=args.hot_pixel_mask,
        generate_set=args.generate_set,
        eval_save_fits=args.save_eval_fits
    )

if __name__ == "__main__":
    main()