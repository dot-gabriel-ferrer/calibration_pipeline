# Author: Elías Gabriel Ferrer Jorge

"""
CLI script to execute the flat pipeline with different modes.
Example usage:
  python run_pipeline.py --basepath /data/flats/ --output-dir /results/flats/ --mode full
"""

import argparse
import json
from main_pipeline import run_flat_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the flat calibration pipeline.")
    parser.add_argument("--basepath", type=str, default="",
                        help="Path to raw flat FITS (required in 'full' or 'reduce_only' modes).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to store all pipeline outputs.")
    parser.add_argument("--mode", type=str, default="full",
                        choices=[
                            "full",
                            "reduce_only",
                            "normalize_only",
                            "vignetting_only",
                            "make_master",
                            "fit_model",
                            "evaluate_model"
                        ],
                        help="Which portion of the pipeline to run.")
    parser.add_argument("--dark-files-json", type=str, default="",
                        help="Path to a JSON file containing dark_files info. If empty, no dark subtraction.")
    parser.add_argument("--norm-method", type=str, default="max", choices=["max","mean"],
                        help="Normalization method for step3 (max or mean).")
    parser.add_argument("--lensfun-json", type=str, default="",
                        help="Path to JSON with lensfun params if you want lensfun correction (step4).")
    parser.add_argument("--radial-json", type=str, default="",
                        help="Path to JSON with radial correction params (step4).")
    parser.add_argument("--empirical-json", type=str, default="",
                        help="Path to JSON with empirical correction table (step4).")
    parser.add_argument("--master-method", type=str, default="median", choices=["median","mean"],
                        help="Combination method in step5 (median or mean).")
    parser.add_argument("--t-ref", type=float, default=0.0,
                        help="Reference temperature for parametric model (step6,7).")
    parser.add_argument("--exp-ref", type=float, default=0.0,
                        help="Reference exposure for parametric model (step6,7).")
    parser.add_argument("--save-eval-fits", action="store_true",
                        help="If set, step7 saves diff/mae/mape as FITS.")
    parser.add_argument("--no-save-plots", action="store_true",
                        help="If set, step7 won't save PNG plots.")
    
    args = parser.parse_args()

    # Load dark_files if provided
    dark_files = None
    if args.dark_files_json:
        with open(args.dark_files_json, 'r') as f:
            dark_files = json.load(f)

    # Load lensfun, radial, empirical parameters if provided
    lensfun_params = None
    if args.lensfun_json:
        with open(args.lensfun_json, 'r') as f:
            lensfun_params = json.load(f)

    radial_params = None
    if args.radial_json:
        with open(args.radial_json, 'r') as f:
            radial_params = json.load(f)

    empirical_params = None
    if args.empirical_json:
        with open(args.empirical_json, 'r') as f:
            empirical_params = json.load(f)
        # If the empirical table is a path to a FITS, load it
        if 'table_path' in empirical_params:
            from astropy.io import fits
            table_data = fits.getdata(empirical_params['table_path'])
            empirical_params['table'] = table_data

    # run pipeline
    run_flat_pipeline(
        basepath=args.basepath,
        output_dir=args.output_dir,
        dark_files=dark_files,
        mode=args.mode,
        norm_method=args.norm_method,
        lensfun_params=lensfun_params,
        radial_params=radial_params,
        empirical_params=empirical_params,
        master_method=args.master_method,
        T_ref=args.t_ref,
        exp_ref=args.exp_ref,
        save_eval_fits=args.save_eval_fits,
        save_eval_plots=(not args.no_save_plots)
    )

if __name__ == "__main__":
    main()
