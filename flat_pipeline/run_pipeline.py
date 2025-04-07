# run_pipeline.py
# Author: Elías Gabriel Ferrer Jorge

"""
Script to execute the flat calibration pipeline.
Supports different modes for modular execution,
and different flat types (sky, lab, satellite).
"""

import argparse
import json
import os

# Importamos la función principal actualizada
from .main_pipeline import run_flat_pipeline

def main():
    parser = argparse.ArgumentParser(description="Flat Calibration Pipeline")

    parser.add_argument(
        "--basepath",
        required=False,
        default="",
        help="Base path where raw FITS are located or intermediate inputs depending on mode"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs will be written"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "full_no_vignetting",
            "reduce_only",
            "normalize_only",
            "vignetting_only",
            "make_master",
            "fit_model",
            "evaluate_model"
        ],
        default="full",
        help="Execution mode"
    )

    # NUEVO: tipo de flat
    parser.add_argument(
        "--flat-type",
        choices=["sky", "lab", "satellite"],
        default="lab",
        help="Type of flats being processed: sky, lab, or satellite."
    )

    parser.add_argument(
        "--dark-files-json",
        default="",
        help="Optional path to JSON list of dark files (only used in reduce/full modes)"
    )

    parser.add_argument(
        "--norm-method",
        choices=["max", "mean"],
        default="max",
        help="Normalization method to use on flats"
    )

    parser.add_argument(
        "--lensfun-json",
        default="",
        help="Path to JSON with lensfun parameters"
    )

    parser.add_argument(
        "--radial-json",
        default="",
        help="Path to JSON with radial correction parameters"
    )

    parser.add_argument(
        "--empirical-json",
        default="",
        help="Path to JSON with empirical correction parameters"
    )

    parser.add_argument(
        "--master-method",
        choices=["median", "mean"],
        default="median",
        help="Method to combine flats into master flats"
    )

    parser.add_argument(
        "--t-ref",
        type=float,
        default=0.0,
        help="Reference temperature used in flat model"
    )

    parser.add_argument(
        "--exp-ref",
        type=float,
        default=0.0,
        help="Reference exposure time used in flat model"
    )

    parser.add_argument(
        "--save-eval-fits",
        action="store_true",
        help="Save evaluation FITS files"
    )

    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Disable saving of evaluation plots"
    )

    args = parser.parse_args()

    # Carga o autogeneración de dark_files (sólo en modos que lo requieran)
    dark_files = None
    if args.mode in ["full", "full_no_vignetting", "reduce_only"]:
        if args.dark_files_json:
            with open(args.dark_files_json, "r") as f:
                dark_files = json.load(f)
        else:
            print("[run_pipeline] Auto-detecting darks from --basepath...")
            from .flat_pipeline.steps.observation_manager.observation_manager import ObservationManager
            manager = ObservationManager(base_path=args.basepath)
            manager.load_and_organize()

            dark_entries = manager.filter_files(
                category='CALIBRATION',
                subcat='darks',
                exp_min=0.1,  # Excluye bias
                ext_filter='fits'
            )

            dark_files = [
                {
                    "original_path": entry["original_path"],
                    "temperature": entry["temperature"],
                    "exposure": entry["exposure"]
                }
                for entry in dark_entries
            ]

            if len(dark_files) == 0:
                raise RuntimeError("No dark frames found automatically in basepath. "
                                   "Please check your FITS structure.")

            auto_json_path = os.path.join(args.output_dir, "auto_dark_files.json")
            with open(auto_json_path, "w") as f:
                json.dump(dark_files, f, indent=2)
            print(f"[run_pipeline] ✅ Found {len(dark_files)} darks. Saved to: {auto_json_path}")

    # Cargar parámetros opcionales de corrección (lensfun, radial, etc.)
    lensfun_params = json.load(open(args.lensfun_json)) if args.lensfun_json else None
    radial_params = json.load(open(args.radial_json)) if args.radial_json else None
    empirical_params = json.load(open(args.empirical_json)) if args.empirical_json else None

    # Ejecutar pipeline principal
    run_flat_pipeline(
        basepath=args.basepath,
        output_dir=args.output_dir,
        dark_files=dark_files,
        mode=args.mode,
        norm_method=args.norm_method,
        lensfun_params=lensfun_params,
        radial_params=radial_params,
        empirical_params=empirical_params,
        master_grouping=('FILTER', 'temperature', 'exposure'),
        master_method=args.master_method,
        T_ref=args.t_ref,
        exp_ref=args.exp_ref,
        save_eval_fits=args.save_eval_fits,
        save_eval_plots=not args.no_save_plots,
        flat_type=args.flat_type  # <--- Nuevo
    )


if __name__ == "__main__":
    main()
