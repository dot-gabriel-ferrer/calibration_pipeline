# Author: Elías Gabriel Ferrer Jorge

"""
Run pipeline script for executing the bias model pipeline end-to-end.
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
