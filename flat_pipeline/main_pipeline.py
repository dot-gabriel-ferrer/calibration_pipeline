# main_pipeline.py
# Autor: Elías Gabriel Ferrer Jorge

import os
from glob import glob
import numpy as np
from astropy.io import fits

from .flat_pipeline.steps.step1_load_data import load_flat_files
from .flat_pipeline.steps.step2_reduce_with_dark import reduce_flats_with_darks
from .flat_pipeline.steps.step3_normalize_flats import normalize_flats_in_dir, normalize_flat
from .flat_pipeline.steps.step4_vignetting_correction import correct_vignetting_in_dir
from .flat_pipeline.steps.step5_make_master_flat import generate_master_flats
from .flat_pipeline.steps.step6_fit_flat_model import run_flat_model_fitting


def run_flat_pipeline(basepath,
                      output_dir,
                      dark_files,
                      mode,
                      norm_method="max",
                      lensfun_params=None,
                      radial_params=None,
                      empirical_params=None,
                      master_grouping=('FILTER', 'temperature', 'exposure'),
                      master_method='median',
                      T_ref=0.0,
                      exp_ref=0.0,
                      save_eval_fits=False,
                      save_eval_plots=True,
                      flat_type="lab"):
    """
    Orquesta los pasos del pipeline, admitiendo distintos 'mode' y 'flat_type'.
    En particular, si flat_type=="sky", se genera un 'master median flat' 
    adicional a partir de los flats reducidos.
    """

    # Paso 0: Crear la estructura de directorios
    print("\n[Step 0: Flat Pipeline] Creating output directory structure...")
    dirs_to_create = [
        output_dir,
        os.path.join(output_dir, "flat_reduced"),
        os.path.join(output_dir, "flat_normalized"),
        os.path.join(output_dir, "flat_vignetting_corrected"),
        os.path.join(output_dir, "master_flat"),
    ]
    # Solo creamos el directorio 'sky_median' si flat_type es sky
    if flat_type == "sky":
        dirs_to_create.append(os.path.join(output_dir, "sky_median"))

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    if mode in [
        "full", "full_no_vignetting", "reduce_only",
        "normalize_only", "vignetting_only", "make_master",
        "fit_model", "evaluate_model"
    ]:

        # --- STEP 1: Cargar flats (si el modo lo requiere) ---
        print("\n[Step 1: Flat Pipeline] Loading flat frames...")
        manager, flat_entries = (None, [])
        if mode in [
            "full", "full_no_vignetting", "reduce_only",
            "normalize_only", "vignetting_only", "make_master",
            "fit_model"
        ]:
            manager, flat_entries = load_flat_files(basepath)
            if len(flat_entries) == 0:
                raise RuntimeError(f"No flat files found in '{basepath}'.")

        # --- STEP 1b: Pre‐procesar (opcional según flat_type) ---
        if flat_type == "sky":
            flat_entries = preprocess_sky_flats(flat_entries)
        elif flat_type == "satellite":
            flat_entries = preprocess_satellite_flats(flat_entries)
        # "lab" no hace nada

        # --- STEP 2: Restar darks ---
        if mode in ["full", "full_no_vignetting", "reduce_only"]:
            print(f"[Step 2] Processing {len(flat_entries)} flats with dark subtraction...")
            reduce_flats_with_darks(
                flat_entries,
                dark_files,
                output_dir=os.path.join(output_dir, "flat_reduced")
            )

        # --- [Step 2b] Generar master median flat específico de sky, si aplica ---
        if flat_type == "sky" and mode in ["full", "full_no_vignetting", "reduce_only"]:
            sky_median_dir = os.path.join(output_dir, "sky_median")
            reduced_dir = os.path.join(output_dir, "flat_reduced")

            # Creamos el master median
            median_sky_path = create_median_sky_flat(
                reduced_dir=reduced_dir,
                sky_median_dir=sky_median_dir
            )

            # Opcional: Normalizar también ese master median (usando la misma lógica de Step3).
            # Si lo deseas, hazlo aquí:
            if median_sky_path is not None and os.path.exists(median_sky_path):
                # Nombre de salida para la versión normalizada
                median_norm_path = os.path.join(
                    sky_median_dir,
                    "master_flat_sky_median_normalized.fits"
                )
                # Usa la misma función 'normalize_flat' de step3
                normalize_flat(median_sky_path, 
                               median_norm_path, 
                               method=norm_method)
                print(f"[Sky] Master median sky flat normalized saved to: {median_norm_path}")

        # --- STEP 3: Normalizar flats ---
        if mode in ["full", "full_no_vignetting", "normalize_only"]:
            print(f"[Step 3] Normalizing by {norm_method.upper()}.")
            normalize_flats_in_dir(
                raw_reduced_dir=os.path.join(output_dir, "flat_reduced"),
                output_dir=os.path.join(output_dir, "flat_normalized"),
                method=norm_method
            )

        # --- STEP 4: Corregir viñeteo ---
        if mode == "full":
            print("[Step 4] Correcting vignetting (full mode).")
            correct_vignetting_in_dir(
                input_dir=os.path.join(output_dir, "flat_normalized"),
                output_dir=os.path.join(output_dir, "flat_vignetting_corrected"),
                lensfun_params=lensfun_params,
                radial_params=radial_params,
                empirical_params=empirical_params
            )
        elif mode == "vignetting_only":
            correct_vignetting_in_dir(
                input_dir=basepath,
                output_dir=os.path.join(output_dir, "flat_vignetting_corrected"),
                lensfun_params=lensfun_params,
                radial_params=radial_params,
                empirical_params=empirical_params
            )

        # --- STEP 5: Generar Master Flats ---
        if mode in ["full", "make_master"]:
            corrected_files = glob(os.path.join(output_dir, "flat_vignetting_corrected", "*.fits"))
            fallback_files = glob(os.path.join(output_dir, "flat_normalized", "*.fits"))

            corrected_entries = build_entries_from_filenames(corrected_files)
            fallback_entries = build_entries_from_filenames(fallback_files)

            generate_master_flats(
                flat_entries=corrected_entries,
                output_dir=os.path.join(output_dir, "master_flat"),
                grouping=master_grouping,
                method=master_method,
                fallback_entries=fallback_entries
            )

        # --- STEP 6: Ajustar modelo de flat ---
        if mode in ["full", "fit_model"]:
            run_flat_model_fitting(
                master_flat_dir=os.path.join(output_dir, "master_flat"),
                T_ref=T_ref,
                exp_ref=exp_ref,
                output_dir=os.path.join(output_dir, "flat_model_fits")
            )

        if mode == "evaluate_model":
            pass

    else:
        raise ValueError(f"Unsupported mode: {mode}")


# ============================================================================
# FUNCIONES DE PRE‐PROCESADO Y UTILIDADES
# ============================================================================
def preprocess_sky_flats(flat_entries):
    print("[Info] Pre-processing sky flats: combining or removing star patterns...")
    # Aquí iría tu lógica real de limpieza de fuentes, etc.
    return flat_entries

def preprocess_satellite_flats(flat_entries):
    print("[Info] Pre-processing satellite flats: cosmic removal, LED pattern correction, etc.")
    return flat_entries

def build_entries_from_filenames(fits_list):
    entries = []
    for f in fits_list:
        entry = {
            "original_path": f,
            "temperature": "UNKNOWN",
            "exposure": "UNKNOWN",
            "FILTER": "UNKNOWN"
        }
        entries.append(entry)
    return entries

def create_median_sky_flat(reduced_dir, sky_median_dir):
    """
    Combina todos los .fits en 'reduced_dir' mediante mediana
    y guarda el resultado en 'sky_median_dir' como master_flat_sky_median.fits.
    Devuelve la ruta al archivo creado o None si no había ficheros.
    """
    pattern = os.path.join(reduced_dir, "*.fits")
    files = glob(pattern)
    if not files:
        print(f"[Sky] No reduced flats found to create median sky flat in {reduced_dir}")
        return None

    stack = []
    for path in files:
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            stack.append(data)

    median_data = np.median(np.stack(stack, axis=0), axis=0)

    out_path = os.path.join(sky_median_dir, "master_flat_sky_median.fits")
    # Cabecera mínima
    h = fits.Header()
    h["SKYPIPE"] = ("median", "Combined sky flats (mediana post-dark-subtraction)")

    # Guardamos
    hdu = fits.PrimaryHDU(data=median_data, header=h)
    hdul_out = fits.HDUList([hdu])
    hdul_out.writeto(out_path, overwrite=True)

    print(f"[Sky] Master median sky flat created: {out_path}")
    return out_path
