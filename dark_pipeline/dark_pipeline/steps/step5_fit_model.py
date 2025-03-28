# step5_fit_model.py
# Author: Elías Gabriel Ferrer Jorge

"""
Step 5: Fit the exponential model DC(T) = A * exp(B * (T - tmin)) for each hot pixel,
and a global A_g, B_g for normal (non-hot) pixels.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def fit_exponential_per_pixel(temps: np.ndarray, pixel_values: np.ndarray, t_min: float) -> tuple:
    """
    Performs a least-squares exponential fit for the model:
    
         DC(T) = A * exp(B * (T - t_min))
    
    In logarithmic space, this becomes:
    
         log(DC) = B * (T - t_min) + log(A)
    
    :param temps: 1D array of temperatures corresponding to the available frames.
    :type temps: np.ndarray
    :param pixel_values: 1D array of measured dark current values for a given pixel at these temperatures.
    :type pixel_values: np.ndarray
    :param t_min: The minimum temperature in the dataset, used to normalize the temperature.
    :type t_min: float
    :return: A tuple (A, B) containing the fitted parameters. If there are not enough positive data points,
             returns (0, 0).
    :rtype: tuple
    """
    # Normalize temperatures
    T_norm = temps - t_min
    
    pixel_values = np.array(pixel_values, dtype=np.float32)
    mask_pos = pixel_values > 0
    if np.count_nonzero(mask_pos) < 2:
        return 0, 0
    
    T_fit = T_norm[mask_pos]
    DC_fit = pixel_values[mask_pos]
    log_DC = np.log(DC_fit)
    A_matrix = np.vstack([T_fit, np.ones(len(T_fit))]).T
    m, c = np.linalg.lstsq(A_matrix, log_DC, rcond=None)[0]
    B = m
    A = np.exp(c)
    return A, B

def model_DC(T: float, A: float, B: float, t_min: float) -> float:
    """
    Exponential dark current model:
         DC(T) = A * exp(B * (T - t_min))
    
    :param T: Temperature in degrees Celsius.
    :type T: float
    :param A: Exponential coefficient.
    :type A: float
    :param B: Exponential growth factor.
    :type B: float
    :param t_min: The minimum temperature used for normalization.
    :type t_min: float
    :return: The dark current at temperature T.
    :rtype: float
    """
    return A * np.exp(B * (T - t_min))

def fit_dark_model(dark_mask_by_temp: dict, model_dir: str) -> tuple:
    """
    Determines hot pixels from a reference temperature mask and fits the parameters
    A and B for each hot pixel using the exponential model:
    
         DC(T) = A * exp(B * (T - t_min))
    
    For normal pixels, it fits global parameters A_g and B_g by averaging the dark current
    across all normal pixels at each temperature.
    
    The minimum temperature (t_min) is automatically computed as the minimum temperature
    among the available dark masks. It is saved along with the model parameters.
    
    :param dark_mask_by_temp: Dictionary mapping temperature to a 2D array of dark current (ADU/s).
    :type dark_mask_by_temp: dict
    :param model_dir: Directory where the outputs (A_map, B_map, hot_pixels, global_params, t_min) will be saved.
    :type model_dir: str
    :return: A tuple containing:
    
        - **A_map**: 2D array of A coefficients.
        - **B_map**: 2D array of B coefficients.
        - **hot_pixels**: 2D boolean array indicating which pixels are classified as hot.
        - **A_g**: Global A coefficient (float) for normal pixels.
        - **B_g**: Global B coefficient (float) for normal pixels.
        - **t_min**: The minimum temperature used for normalization.
    :rtype: tuple
    """
    # Sort masks by temperature
    sorted_masks = sorted(dark_mask_by_temp.items(), key=lambda x: x[0])
    ref_temp, ref_mask = sorted_masks[0]
    
    # Compute t_min as the minimum temperature from the dataset
    t_min = min(dark_mask_by_temp.keys())
    print(f"Computed t_min = {t_min}")
    
    # Compute threshold to define hot pixels
    median_ref = np.mean(ref_mask)
    std_ref = np.std(ref_mask)
    threshold = median_ref - 5 * std_ref
    
    hot_pixels = (ref_mask > threshold)
    shape = ref_mask.shape
    
    A_map = np.zeros(shape, dtype=np.float32)
    B_map = np.zeros(shape, dtype=np.float32)
    
    all_temps = np.array([t for (t, _) in sorted_masks], dtype=np.float32)
    
    # Fit A, B per hot pixel using normalized temperatures (T - t_min)
    for y in tqdm(range(shape[0]), desc="Fitting A,B per pixel"):
        for x in range(shape[1]):
            if not hot_pixels[y, x]:
                continue
            pixel_values = []
            for temp, mask_avg in sorted_masks:
                pixel_values.append(mask_avg[y, x])
            A_, B_ = fit_exponential_per_pixel(all_temps, np.array(pixel_values), t_min)
            A_map[y, x] = A_
            B_map[y, x] = B_
    
    # Fit global A_g, B_g for normal (non-hot) pixels
    mean_dc_each_temp = []
    for temp, mask_avg in sorted_masks:
        mean_dc_each_temp.append(np.mean(mask_avg[~hot_pixels]))
    A_g, B_g = fit_exponential_per_pixel(all_temps, np.array(mean_dc_each_temp), t_min)
    print(f"Global normal-pixels fit => A_g={A_g:.4e}, B_g={B_g:.4e}")
    
    # Save A_map, B_map, hot_pixels as FITS and global parameters (including t_min)
    os.makedirs(model_dir, exist_ok=True)
    out_A_map = os.path.join(model_dir, "A_map.fits")
    out_B_map = os.path.join(model_dir, "B_map.fits")
    out_hot = os.path.join(model_dir, "hot_pixels.fits")
    
    fits.PrimaryHDU(A_map).writeto(out_A_map, overwrite=True)
    fits.PrimaryHDU(B_map).writeto(out_B_map, overwrite=True)
    fits.PrimaryHDU(hot_pixels.astype(np.uint8)).writeto(out_hot, overwrite=True)
    
    # Save global parameters including t_min
    global_params_path = os.path.join(model_dir, "global_params.npz")
    np.savez(global_params_path, A_g=A_g, B_g=B_g, t_min=t_min)
    print(f"A_map, B_map, hot_pixels, global_params (with t_min) saved in {model_dir}")
    
    return A_map, B_map, hot_pixels, A_g, B_g, t_min

def fit_dark_model_by_exposure(masks_by_exp_and_temp: dict, model_dir: str) -> dict:
    """
    Ajusta el modelo exponencial DC(T) = A * exp(B * (T - t_min)) para cada exposure_time,
    guardando los resultados en subcarpetas por exposición.

    :param masks_by_exp_and_temp: dict[exposure_time][temperature] -> dark mask (ADU/s)
    :type masks_by_exp_and_temp: dict
    :param model_dir: Directorio base donde se guardarán los modelos por exposición
    :type model_dir: str
    :return: dict[exposure_time] -> (A_map, B_map, hot_pixels, A_g, B_g, t_min)
    :rtype: dict
    """
    results = {}

    for exposure_time, dark_mask_by_temp in masks_by_exp_and_temp.items():
        print(f"\n[🛠️] Fitting model for exposure time: {exposure_time:.2f}s")

        subdir = os.path.join(model_dir, f"exp_{exposure_time:.2f}".replace('.', 'p'))
        res = fit_dark_model(dark_mask_by_temp, subdir)
        results[exposure_time] = res

    return results
