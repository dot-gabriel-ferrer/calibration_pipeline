# step7_generate_only.py
# Author: Elías Gabriel Ferrer Jorge

"""
Step 7: Load a previously fitted dark model (A_map, B_map, hot_pixels, A_g, B_g)
and generate synthetic dark frames at user-specified temperatures.
"""

import os
import numpy as np
from astropy.io import fits
from .step6_generate_synthetics import generate_precise_synthetic_dark, model_DC


def load_model_and_generate_synthetic(model_dir: str, output_dir: str, temps_str: str):
    """
    Loads the saved dark model files from ``model_dir`` and generates synthetic dark frames.
    
    The following files are loaded from ``model_dir``:
      - ``A_map.fits``
      - ``B_map.fits``
      - ``hot_pixels.fits``
      - ``global_params.npz`` (which includes global parameters A_g, B_g, and t_min)
    
    The function parses the string ``temps_str`` into a list of temperatures and then generates 
    one synthetic dark frame for each temperature using the normalized model with T_norm = T - t_min.
    The resulting synthetic frames are saved as FITS files in the directory specified by ``output_dir``.
    
    :param model_dir: Directory containing ``A_map.fits``, ``B_map.fits``, ``hot_pixels.fits``, and ``global_params.npz``.
    :type model_dir: str
    :param output_dir: Destination directory for the generated synthetic FITS files.
    :type output_dir: str
    :param temps_str: A space-separated string of temperature values (e.g., "20.0 25.0 30.5").
    :type temps_str: str
    """
    os.makedirs(output_dir, exist_ok=True)
    temps_list = [float(t) for t in temps_str.split()]
    
    path_A_map = os.path.join(model_dir, "A_map.fits")
    path_B_map = os.path.join(model_dir, "B_map.fits")
    path_hotpix = os.path.join(model_dir, "hot_pixels.fits")
    path_global = os.path.join(model_dir, "global_params.npz")
    
    with fits.open(path_A_map) as hdul:
        A_map = hdul[0].data.astype(np.float32)
    with fits.open(path_B_map) as hdul:
        B_map = hdul[0].data.astype(np.float32)
    with fits.open(path_hotpix) as hdul:
        hot_pixels = hdul[0].data.astype(bool)
    
    data = np.load(path_global)
    A_g = float(data["A_g"])
    B_g = float(data["B_g"])
    t_min = float(data["t_min"])
    
    print(f"\nLoaded model from '{model_dir}':")
    print(f"  A_map shape = {A_map.shape}, B_map shape = {B_map.shape}")
    print(f"  Global parameters: A_g={A_g:.4e}, B_g={B_g:.4e}, t_min={t_min:.2f}")
    
    for temp in temps_list:
        synthetic_frame = generate_precise_synthetic_dark(
            T_new=temp,
            A_map=A_map,
            B_map=B_map,
            hot_pixel_mask=hot_pixels,
            A_g=A_g,
            B_g=B_g,
            t_min=t_min,
            shape=A_map.shape
        )
        
        hdr = fits.Header()
        hdr["COMMENT"] = "Synthetic dark frame from loaded model."
        hdr["TEMP"] = temp
        hdr["T_MIN"] = t_min
        hdr["A_G"] = A_g
        hdr["B_G"] = B_g
        hdr["BUNIT"] = "ADU/s"
        
        hdu = fits.PrimaryHDU(synthetic_frame, header=hdr)
        out_name = f"synthetic_dark_{temp:.2f}.fits"
        out_path = os.path.join(output_dir, out_name)
        hdu.writeto(out_path, overwrite=True)
        print(f"Saved synthetic dark at T={temp:.2f}°C -> {out_path}")

def generate_synthetics_by_exposure(base_model_dir: str, output_dir: str, temps_str: str):
    """
    Genera sintéticos para múltiples modelos ajustados por exposure_time,
    leyendo desde subdirectorios del modelo base.

    :param base_model_dir: Directorio base donde están los modelos por exposición.
    :param output_dir: Directorio donde guardar los FITS generados.
    :param temps_str: Cadena de temperaturas separadas por espacios.
    """
    os.makedirs(output_dir, exist_ok=True)
    temps_list = [float(t) for t in temps_str.split()]

    for subdir_name in sorted(os.listdir(base_model_dir)):
        if not subdir_name.startswith("exp_"):
            continue
        
        exp_subdir = os.path.join(base_model_dir, subdir_name)
        if not os.path.isdir(exp_subdir):
            continue

        try:
            path_A_map = os.path.join(exp_subdir, "A_map.fits")
            path_B_map = os.path.join(exp_subdir, "B_map.fits")
            path_hotpix = os.path.join(exp_subdir, "hot_pixels.fits")
            path_global = os.path.join(exp_subdir, "global_params.npz")

            with fits.open(path_A_map) as hdul:
                A_map = hdul[0].data.astype(np.float32)
            with fits.open(path_B_map) as hdul:
                B_map = hdul[0].data.astype(np.float32)
            with fits.open(path_hotpix) as hdul:
                hot_pixels = hdul[0].data.astype(bool)
            data = np.load(path_global)
            A_g = float(data["A_g"])
            B_g = float(data["B_g"])
            t_min = float(data["t_min"])
            shape = A_map.shape
        except Exception as e:
            print(f"[!] Error loading model from {exp_subdir}: {e}")
            continue

        print(f"\nLoaded model from '{subdir_name}': A_g={A_g:.4e}, B_g={B_g:.4e}, t_min={t_min:.2f}")

        output_subdir = os.path.join(output_dir, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)

        for temp in temps_list:
            synthetic_frame = generate_precise_synthetic_dark(
                T_new=temp,
                A_map=A_map,
                B_map=B_map,
                hot_pixel_mask=hot_pixels,
                A_g=A_g,
                B_g=B_g,
                t_min=t_min,
                shape=shape
            )

            hdr = fits.Header()
            hdr["COMMENT"] = "Synthetic dark frame from loaded model."
            hdr["TEMP"] = temp
            hdr["T_MIN"] = t_min
            hdr["A_G"] = A_g
            hdr["B_G"] = B_g
            hdr["BUNIT"] = "ADU/s"

            out_name = f"synthetic_dark_{temp:.2f}.fits"
            out_path = os.path.join(output_subdir, out_name)
            fits.PrimaryHDU(synthetic_frame, header=hdr).writeto(out_path, overwrite=True)
            print(f"Saved T={temp:.2f}°C -> {out_path}")
