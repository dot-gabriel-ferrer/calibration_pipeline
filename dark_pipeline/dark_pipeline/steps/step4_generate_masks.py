# Author: Elías Gabriel Ferrer Jorge

"""
Step 4: Generate dark current masks by averaging the bias-corrected dark frames,
grouped by temperature, and normalizing by exposure time (ADU/s).
"""

import os
import numpy as np
from astropy.io import fits


def generate_dark_masks(corrected_dark_frames: list, output_dir: str) -> dict:
    """
    Generates average dark masks (in ADU/s) grouped by temperature from the
    bias-corrected dark frames. The resulting 2D arrays are written to disk as FITS
    and also returned in a dictionary.

    :param corrected_dark_frames: List of dict objects containing information about
                                  the bias-corrected dark frames (temperature, exposure, data, etc.).
    :type corrected_dark_frames: list
    :param output_dir: Path to the folder where the averaged dark masks will be saved.
    :type output_dir: str
    :return: A dictionary mapping temperature -> averaged dark mask (2D numpy array, ADU/s).
    :rtype: dict
    """
    dark_current_by_temp = {}
    # Group corrected frames by temperature
    for item in corrected_dark_frames:
        temp = item['temperature']
        exp = item['exposure']
        dc_frame = item['corrected_data'] / exp  # Convert to ADU/s
        dark_current_by_temp.setdefault(temp, []).append(dc_frame)

    dark_mask_by_temp = {}
    # Compute average mask per temperature
    for temp, frames in dark_current_by_temp.items():
        stack = np.stack(frames, axis=0)
        mask_avg = np.mean(stack, axis=0)

        # Example of simple cleanup (avoid negative or extremely low values)
        M = np.median(mask_avg)
        S = np.std(mask_avg)
        floor = M - S

        # Clip the mask to remove values below a threshold
        mask_avg = np.clip(mask_avg, a_min=M, a_max=None)
        mask_avg = np.where(mask_avg < floor, floor, mask_avg)
        if floor < 0:
            mask_avg -= floor

        dark_mask_by_temp[temp] = mask_avg

        # Write to FITS
        hdr = fits.Header()
        hdr['BUNIT'] = 'ADU/s'
        hdr['TEMP'] = temp
        hdr['COMMENT'] = "Dark current mask (bias-corrected)"
        hdr['HISTORY'] = "Average of corrected dark frames, normalized by exposure"

        out_name = f"dark_mask_{temp:.2f}.fits"
        out_path = os.path.join(output_dir, out_name)
        fits.PrimaryHDU(data=mask_avg, header=hdr).writeto(out_path, overwrite=True)
        print(f"Saved dark mask for temp={temp:.2f} -> {out_path}")

    return dark_mask_by_temp


def generate_dark_masks_by_exposure(corrected_by_exposure: dict, output_dir: str) -> dict:
    """
    Generates dark current masks (in ADU/s) for each exposure time and temperature group,
    averaging the corrected dark frames and normalizing by exposure time.

    :param corrected_by_exposure: dict[exposure_time] -> list of corrected dark frames (con 'corrected_data', etc.)
    :type corrected_by_exposure: dict
    :param output_dir: Base directory to save all output dark masks
    :type output_dir: str
    :return: dict[exposure_time][temperature] -> dark mask (2D numpy array, in ADU/s)
    :rtype: dict
    """
    masks_by_exp_and_temp = {}

    for exposure_time, frame_list in corrected_by_exposure.items():
        exp_str = f"{exposure_time:.2f}".replace('.', 'p')
        subdir = os.path.join(output_dir, f"exp_{exp_str}")
        os.makedirs(subdir, exist_ok=True)

        # Agrupar por temperatura
        dark_by_temp = {}
        for item in frame_list:
            temp = item['temperature']
            dc_frame = item['corrected_data'] / exposure_time  # Convert to ADU/s
            dark_by_temp.setdefault(temp, []).append(dc_frame)

        masks_by_temp = {}
        for temp, frames in dark_by_temp.items():
            stack = np.stack(frames, axis=0)
            mask_avg = np.mean(stack, axis=0)

            # Opcional: limpieza básica
            M = np.median(mask_avg)
            S = np.std(mask_avg)
            floor = M - S
            mask_avg = np.clip(mask_avg, a_min=M, a_max=None)
            mask_avg = np.where(mask_avg < floor, floor, mask_avg)
            if floor < 0:
                mask_avg -= floor

            masks_by_temp[temp] = mask_avg

            # Guardar FITS
            hdr = fits.Header()
            hdr['BUNIT'] = 'ADU/s'
            hdr['TEMP'] = temp
            hdr['EXPTIME'] = exposure_time
            hdr['COMMENT'] = "Dark current mask (bias-corrected)"
            hdr['HISTORY'] = "Average of corrected dark frames, normalized by exposure"

            out_name = f"dark_mask_t{temp:.2f}_e{exposure_time:.2f}.fits"
            out_path = os.path.join(subdir, out_name)
            fits.PrimaryHDU(data=mask_avg, header=hdr).writeto(out_path, overwrite=True)
            print(f"[✓] Saved mask T={temp:.2f} | Exp={exposure_time:.2f} -> {out_path}")

        masks_by_exp_and_temp[exposure_time] = masks_by_temp

    return masks_by_exp_and_temp
