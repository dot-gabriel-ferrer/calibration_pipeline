# main_pipeline.py
# Author: Elías Gabriel Ferrer Jorge

"""
Main coordination module for the dark-frame calibration pipeline,
now fully supporting exposure_time grouping for per-exposure analysis.
"""

import os
from typing import Optional

from dark_pipeline.steps.step0_paths import create_directories
from dark_pipeline.steps.step1_load_data import load_observations
from dark_pipeline.steps.step2_generate_bias import generate_bias_maps, check_bias_frames
from dark_pipeline.steps.step3_subtract_bias import subtract_bias_grouped_by_exposure
from dark_pipeline.steps.step4_generate_masks import generate_dark_masks_by_exposure
from dark_pipeline.steps.step5_fit_model import fit_dark_model_by_exposure
from dark_pipeline.steps.step6_generate_synthetics import evaluate_models_by_exposure
from dark_pipeline.steps.step7_generate_only import generate_synthetics_by_exposure
from dark_pipeline.steps.step8_evaluate_model import evaluate_all_models
from dark_pipeline.steps.step9_evaluate_dark_current import evaluate_dark_current_all_exposures
from dark_pipeline.steps.step10_fit_2D_model import fit_dark_model_2D
from dark_pipeline.steps.step11_evaluate_model_2D import evaluate_model_2D
from dark_pipeline.steps.step12_generate_synthetic_2D import generate_synthetic_2D
from dark_pipeline.steps.step12_generate_synthetic_2D import generate_single_synthetic_2D

def run_full_pipeline(
    basepath: str,
    output_dir: str,
    mode: str = "full",
    model_dir: str = "",
    temps: str = "20.0",
    temp_single: float = 20.0,
    exptime_single: float = 10.0
) -> None:
    if mode == "full":
        dirs = create_directories(output_dir)

        manager, grouped_darks, short_darks = load_observations(basepath)
        print(f"Found {sum(len(v) for v in grouped_darks.values())} long darks and {len(short_darks)} short darks (bias).")

        good_bias, _ = check_bias_frames(short_darks, log_path='suspicious_bias.log')
        bias_map_by_temp = generate_bias_maps(good_bias)

        corrected_by_exposure = subtract_bias_grouped_by_exposure(
            grouped_darks,
            bias_map_by_temp,
            dirs['dark_corrected_dir']
        )

        masks_by_exp_and_temp = generate_dark_masks_by_exposure(
            corrected_by_exposure,
            dirs['dark_masks_dir']
        )

        model_results = fit_dark_model_by_exposure(
            masks_by_exp_and_temp,
            dirs['model_dir']
        )

        evaluate_models_by_exposure(
            masks_by_exp_and_temp,
            model_results
        )

        # --- NEW: Fit and Evaluate 2D Model ---
        masks_root = os.path.join(output_dir, "dark_masks_by_temp")
        hot_pixel_mask_path = os.path.join(output_dir, "dark_model", "exp_1p00", "hot_pixels.fits")
        fit_dark_model_2D(masks_root, hot_pixel_mask_path, os.path.join(output_dir, "dark_model_2d"))
        evaluate_model_2D(os.path.join(output_dir, "dark_model_2d"), masks_root, output_dir)

        print("\nPipeline successfully completed (mode=full).")
        print(f"All outputs are stored in: {output_dir}")

    elif mode == "synthetic_only":
        if not model_dir:
            raise ValueError("--model_dir is required in synthetic_only mode.")

        generate_synthetics_by_exposure(
            base_model_dir=model_dir,
            output_dir=output_dir,
            temps_str=temps
        )

        print("\nSynthetic generation completed (mode=synthetic_only).")

    elif mode == "evaluation_only":
        if not model_dir:
            model_dir = os.path.join(output_dir, "dark_model")

        candidate_dirs = ["dark_masks_dir", "dark_masks_by_temp"]
        mask_dir = None
        for cand in candidate_dirs:
            path = os.path.join(output_dir, cand)
            if os.path.isdir(path):
                mask_dir = path
                break

        if mask_dir is None:
            raise FileNotFoundError("No mask directory found. Expected one of: dark_masks_dir, dark_masks_by_temp.")

        evaluate_all_models(model_dir, mask_dir, output_dir)

        print("\nModel evaluation completed (mode=evaluation_only).")

    elif mode == "evaluate_dark_current":
        candidate_dirs = ["dark_masks_dir", "dark_masks_by_temp"]
        mask_dir = None
        for cand in candidate_dirs:
            path = os.path.join(output_dir, cand)
            if os.path.isdir(path):
                mask_dir = path
                break

        if mask_dir is None:
            raise FileNotFoundError("No mask directory found. Expected one of: dark_masks_dir, dark_masks_by_temp.")

        evaluate_dark_current_all_exposures(mask_dir, output_dir)

        print("\nDark current analysis completed (mode=evaluate_dark_current).")

    elif mode == "fit_2d_model":
        masks_root = os.path.join(output_dir, "dark_masks_by_temp")
        hot_pixel_mask_path = os.path.join(output_dir, "dark_model", "exp_1p00", "hot_pixels.fits")
        fit_dark_model_2D(masks_root, hot_pixel_mask_path, os.path.join(output_dir, "dark_model_2d"))

    elif mode == "evaluate_2d_model":
        masks_root = os.path.join(output_dir, "dark_masks_by_temp")
        model_dir = os.path.join(output_dir, "dark_model_2d")
        evaluate_model_2D(model_dir, masks_root, output_dir)

    elif mode == "synthetic_2d":
        model_dir = os.path.join(output_dir, "dark_model_2d")
        generate_synthetic_2D(model_dir, output_dir, temps)

    elif mode == "synthetic_2d_single":
        model_dir = os.path.join(output_dir, "dark_model_2d")
        output_path = os.path.join(output_dir, f"synthetic_2d_dark_T{temp_single:.2f}_E{exptime_single:.2f}.fits")
        generate_single_synthetic_2D(temp_single, exptime_single, model_dir, output_path)


    else:
        raise ValueError(f"Invalid mode: {mode}.")


def find_first_hot_pixel_mask(model_dir):
    for sub in sorted(os.listdir(model_dir)):
        subdir = os.path.join(model_dir, sub)
        candidate = os.path.join(subdir, "hot_pixels.fits")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("No 'hot_pixels.fits' found in any subdirectory of dark_model.")

def run_fit_2d_model(output_dir: str, max_files_per_pixel: int = 50):
    print("[•] Ajustando modelo 2D por píxel...")
    model_dir = os.path.join(output_dir, "dark_model_2D")
    masks_root = os.path.join(output_dir, "dark_masks_by_temp")
    hot_pixel_mask_path = find_first_hot_pixel_mask(os.path.join(output_dir, "dark_model"))

    fit_dark_model_2D(
        masks_root=masks_root,
        hot_pixel_mask_path=hot_pixel_mask_path,
        output_dir=model_dir,
        max_files_per_pixel=max_files_per_pixel
    )

