# run_pipeline.py
# Author: Elías Gabriel Ferrer Jorge

"""
Script to execute the dark current calibration pipeline.
Supports full processing, synthetic generation, model evaluation,
and now 2D model generation and prediction.
"""

import argparse
from dark_pipeline.main_pipeline import run_full_pipeline, run_fit_2d_model


def main():
    parser = argparse.ArgumentParser(description="Dark Frame Calibration Pipeline")

    parser.add_argument(
        "--basepath",
        required=False,
        default="",
        help="Base path with FITS files (needed for full mode)"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for saving results (required)"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "synthetic_only",
            "evaluation_only",
            "evaluate_dark_current",
            "fit_2d_model",
            "evaluate_2d_model",
            "synthetic_2d",
            "synthetic_2d_single"
        ],
        default="full",
        help="Pipeline mode"
    )

    parser.add_argument(
        "--model_dir",
        default="",
        help="Directory with model files (required in synthetic_only mode)"
    )

    parser.add_argument(
        "--temps",
        default="20.0",
        help="Temperatures for synthetic generation (e.g. '15.0 20.0 25.0')"
    )

    parser.add_argument(
    "--temp_single",
    type=float,
    default=20.0,
    help="Temperature for single 2D synthetic generation"
    )

    parser.add_argument(
        "--exptime_single",
        type=float,
        default=10.0,
        help="Exposure time for single 2D synthetic generation"
    )


    args = parser.parse_args()

    run_full_pipeline(
        basepath=args.basepath,
        output_dir=args.output_dir,
        mode=args.mode,
        model_dir=args.model_dir,
        temps=args.temps
    )


if __name__ == "__main__":
    main()
