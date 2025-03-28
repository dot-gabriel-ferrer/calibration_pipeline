# Author: Elías Gabriel Ferrer Jorge

"""
Step 6: Generate synthetic dark frames based on the fitted exponential model 
and evaluate the Mean Absolute Error (MAE) compared to real dark masks.
"""

import numpy as np


def model_DC(T: float, A: float, B: float, t_min: float) -> float:    
    """
    Computes DC(T) = A * exp(B*T-T_min).

    :param T: Temperature at which to compute the dark current.
    :type T: float
    :param A: Exponential coefficient.
    :type A: float
    :param B: Temperature exponent factor.
    :type B: float
    :return: Dark current at temperature T.
    :rtype: float
    """
    return A * np.exp(B * (T - t_min))


def generate_precise_synthetic_dark(
    T_new: float,
    A_map: np.ndarray,
    B_map: np.ndarray,
    hot_pixel_mask: np.ndarray,
    A_g: float,
    B_g: float,
    t_min: float,
    shape: tuple
) -> np.ndarray:
    """
    Generates a synthetic dark frame for temperature T_new using:
      - Per-pixel model for hot pixels: DC = A_map[y,x] * exp(B_map[y,x] * (T_new - t_min))
      - Global model (A_g, B_g) for normal pixels.
    
    :param T_new: Desired temperature.
    :param A_map: 2D array of A coefficients.
    :param B_map: 2D array of B coefficients.
    :param hot_pixel_mask: 2D boolean array for hot pixels.
    :param A_g: Global A coefficient.
    :param B_g: Global B coefficient.
    :param t_min: Minimum temperature used for normalization.
    :param shape: Shape of the output frame.
    :return: Synthetic dark frame as a 2D numpy array.
    """
    synthetic = np.zeros(shape, dtype=np.float32)
    global_dark_value = model_DC(T_new, A_g, B_g, t_min)
    
    valid_dark_pixels = (hot_pixel_mask) & (A_map > 0) & (B_map != 0)
    synthetic[valid_dark_pixels] = model_DC(T_new, A_map[valid_dark_pixels], B_map[valid_dark_pixels], t_min)
    synthetic[~valid_dark_pixels] = global_dark_value
    return synthetic


def generate_synthetics_and_evaluate(
    dark_mask_by_temp: dict,
    A_map: np.ndarray,
    B_map: np.ndarray,
    hot_pixels: np.ndarray,
    A_g: float,
    B_g: float,
    t_min:float
) -> None:
    """
    Generates synthetic dark frames for each temperature present in dark_mask_by_temp,
    then computes and prints the Mean Absolute Error (MAE) between the synthesized frame
    and the actual dark mask.

    :param dark_mask_by_temp: Dictionary {temp: 2D real dark mask} for each temperature.
    :type dark_mask_by_temp: dict
    :param A_map: 2D array of A coefficients for hot pixels.
    :type A_map: np.ndarray
    :param B_map: 2D array of B coefficients for hot pixels.
    :type B_map: np.ndarray
    :param hot_pixels: 2D boolean array indicating hot pixels.
    :type hot_pixels: np.ndarray
    :param A_g: Global A coefficient for normal pixels.
    :type A_g: float
    :param B_g: Global B coefficient for normal pixels.
    :type B_g: float
    """
    # Retrieve shape from any mask in the dictionary
    shape = next(iter(dark_mask_by_temp.values())).shape

    mae_results = []
    # Evaluate each temperature
    for (temp, real_mask) in sorted(dark_mask_by_temp.items(), key=lambda x: x[0]):
        synth_dark = generate_precise_synthetic_dark(
            T_new=temp,
            A_map=A_map,
            B_map=B_map,
            hot_pixel_mask=hot_pixels,
            A_g=A_g,
            B_g=B_g,
            shape=shape,
            t_min=t_min
        )

        diff = synth_dark - real_mask
        mae = np.mean(np.abs(diff)) #/ 16
        mae_results.append((temp, mae)) #/16 

    # Sort by temperature and print results
    mae_results.sort(key=lambda x: x[0])
    print("MAE by temperature:")
    for t, mae in mae_results:
        print(f" T={t:.2f}°C => MAE={mae:.4f} ADU/s (12 scaled to 16 bits)")


def evaluate_models_by_exposure(
    masks_by_exp_and_temp: dict,
    model_results: dict
):
    """
    Evalúa los modelos ajustados comparando los dark masks reales y sintéticos
    para cada exposure_time, mostrando el MAE por temperatura.

    :param masks_by_exp_and_temp: dict[exposure_time][temperature] -> dark mask
    :type masks_by_exp_and_temp: dict
    :param model_results: dict[exposure_time] -> (A_map, B_map, hot_pixels, A_g, B_g, t_min)
    :type model_results: dict
    """
    for exposure_time, dark_mask_by_temp in masks_by_exp_and_temp.items():
        print(f"\n[🔍] Evaluating model for exposure time {exposure_time:.2f}s")
        
        A_map, B_map, hot_pixels, A_g, B_g, t_min = model_results[exposure_time]
        shape = next(iter(dark_mask_by_temp.values())).shape

        mae_results = []

        for (temp, real_mask) in sorted(dark_mask_by_temp.items(), key=lambda x: x[0]):
            synth_dark = generate_precise_synthetic_dark(
                T_new=temp,
                A_map=A_map,
                B_map=B_map,
                hot_pixel_mask=hot_pixels,
                A_g=A_g,
                B_g=B_g,
                shape=shape,
                t_min=t_min
            )

            diff = synth_dark - real_mask
            mae = np.mean(np.abs(diff))
            mae_results.append((temp, mae))

        print("  MAE by temperature:")
        for t, mae in mae_results:
            print(f"    T={t:.2f}°C => MAE={mae:.4f} ADU/s")


