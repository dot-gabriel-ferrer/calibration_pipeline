# step12_generate_synthetic_2D.py
# Author: Elías Gabriel Ferrer Jorge

"""
Generate synthetic dark frames using the 2D model:
DC(T, t_exp) = A * t_exp^gamma * exp(B * (T - t_min))
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def model_2D(T: float, t_exp: float, A: float, B: float, gamma: float, t_min: float) -> float:
    return A * (t_exp ** gamma) * np.exp(B * (T - t_min))

def generate_single_frame_vectorized(A_map, B_map, GAMMA_map, T: float, t_exp: float, t_min: float) -> np.ndarray:
    frame = A_map * np.power(t_exp, GAMMA_map) * np.exp(B_map * (T - t_min))
    return frame.astype(np.float32, copy=False)


def generate_synthetic_frame_2D(T: float, t_exp: float,
                                 A_map: np.ndarray,
                                 B_map: np.ndarray,
                                 GAMMA_map: np.ndarray,
                                 t_min: float,
                                 shape: tuple) -> np.ndarray:
    """
    Generate a synthetic dark current frame using the 2D model:
        DC(T, t) = A * t^gamma * exp(B * (T - t_min))

    :param T: Temperature [°C]
    :param t_exp: Exposure time [s]
    :param A_map: Map of A coefficients (per pixel)
    :param B_map: Map of B coefficients (per pixel)
    :param GAMMA_map: Map of gamma coefficients (per pixel)
    :param t_min: Minimum temperature used for normalization
    :param shape: Output frame shape (usually same as maps)
    :return: 2D synthetic dark frame [ADU/s]
    """
    T_delta = T - t_min
    with np.errstate(over='ignore', invalid='ignore'):
        synthetic = A_map * (t_exp ** GAMMA_map) * np.exp(B_map * T_delta)
    synthetic = np.nan_to_num(synthetic, nan=0.0, posinf=0.0, neginf=0.0)
    return synthetic.reshape(shape)



def generate_synthetic_2D(model_dir: str, output_dir: str, temps_str: str):
    os.makedirs(output_dir, exist_ok=True)
    A_map = fits.getdata(os.path.join(model_dir, "A_map_2D.fits"))
    B_map = fits.getdata(os.path.join(model_dir, "B_map_2D.fits"))
    GAMMA_map = fits.getdata(os.path.join(model_dir, "GAMMA_map_2D.fits"))
    t_min = float(np.load(os.path.join(model_dir, "global_params_2d.npz"))["t_min"])

    temperatures = [float(t) for t in temps_str.split()]
    exposure_times = [1.0, 5.0, 10.0, 20.0, 25.0]

    for t_exp in exposure_times:
        out_subdir = os.path.join(output_dir, f"synthetic_2d_exp_{t_exp:.2f}")
        os.makedirs(out_subdir, exist_ok=True)

        for T in tqdm(temperatures, desc=f"[exp={t_exp:.2f}] Generating synthetic frames", leave=False):
            frame = generate_single_frame_vectorized(A_map, B_map, GAMMA_map, T, t_exp, t_min)

            hdr = fits.Header()
            hdr["COMMENT"] = "Synthetic dark frame from 2D model."
            hdr["TEMP"] = T
            hdr["EXPTIME"] = t_exp
            hdr["T_MIN"] = t_min
            hdr["BUNIT"] = "ADU/s"

            hdu = fits.PrimaryHDU(frame, header=hdr)
            out_name = f"synthetic_2d_dark_t{T:.2f}_e{t_exp:.2f}.fits"
            hdu.writeto(os.path.join(out_subdir, out_name), overwrite=True)

        print(f"[✓] Saved synthetic 2D frames for exposure={t_exp:.2f} s -> {out_subdir}")

def generate_single_synthetic_2D(T: float, t_exp: float, model_dir: str, output_path: str):
    A_map = fits.getdata(os.path.join(model_dir, "A_map_2D.fits")).astype(np.float32)
    B_map = fits.getdata(os.path.join(model_dir, "B_map_2D.fits")).astype(np.float32)
    GAMMA_map = fits.getdata(os.path.join(model_dir, "GAMMA_map_2D.fits")).astype(np.float32)
    t_min = float(np.load(os.path.join(model_dir, "global_params_2d.npz"))["t_min"])

    frame = generate_single_frame_vectorized(A_map, B_map, GAMMA_map, T, t_exp, t_min)

    hdr = fits.Header()
    hdr["COMMENT"] = "Synthetic dark frame from 2D model."
    hdr["TEMP"] = float(T)
    hdr["EXPTIME"] = float(t_exp)
    hdr["T_MIN"] = float(t_min)
    hdr["BUNIT"] = "ADU/s"

    hdu = fits.PrimaryHDU(data=frame, header=hdr)
    hdu.writeto(
        output_path,
        overwrite=True,
        output_verify="ignore"
    )

    print(f"[✓] Saved synthetic 2D dark frame T={T:.2f}, exp={t_exp:.2f} -> {output_path}")
