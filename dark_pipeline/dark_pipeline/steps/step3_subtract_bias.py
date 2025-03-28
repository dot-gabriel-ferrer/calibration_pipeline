# Author: Elías Gabriel Ferrer Jorge

"""
Step 3: Subtract the bias from long dark frames using the nearest-temperature bias map
and save the corrected frames in FITS format.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from .utils.utils_scaling import load_fits_scaled_12bit


def get_nearest_bias_map(temp: float, bias_maps: dict) -> np.ndarray:
    """
    Returns the bias map that corresponds to the temperature closest to 'temp'.

    :param temp: The target temperature for which a bias map is needed.
    :type temp: float
    :param bias_maps: Dictionary of temperature -> bias_map.
    :type bias_maps: dict
    :return: The 2D numpy array representing the bias map for the nearest temperature.
    :rtype: np.ndarray
    """
    temps_array = np.array(list(bias_maps.keys()))
    idx_min = np.argmin(np.abs(temps_array - temp))
    nearest_temp = temps_array[idx_min]
    return bias_maps[nearest_temp]


def subtract_bias_from_darks(
    long_darks: list,
    bias_map_by_temp: dict,
    output_dir: str
    ) -> list:
    """
    Subtracts the nearest matching bias map from each long dark frame and saves
    the result in a FITS file with two HDUs:
    
      1) Primary image (bias-corrected data)
      2) Secondary image named 'BIAS_MAP' (the bias map used)

    :param long_darks: List of dictionaries describing long dark frames (including file paths, temperature, exposure, etc.).
    :type long_darks: list
    :param bias_map_by_temp: Dictionary mapping temperature to the averaged bias map (2D array).
    :type bias_map_by_temp: dict
    :param output_dir: Path where the bias-corrected FITS files will be written.
    :type output_dir: str
    :return: A list of dictionaries containing metadata for each corrected dark frame. Each dictionary includes:
    
      - ``original_path``: the path to the original file.
      - ``corrected_path``: the path to the bias-corrected FITS file.
      - ``temperature``: the temperature of the dark frame.
      - ``exposure``: the exposure time.
      - ``corrected_data``: the bias-corrected data array.
      - ``bias_map``: the bias map used for the correction.
    :rtype: list
    """
    corrected_dark_frames = []
    for idx, dark in enumerate(
        tqdm(long_darks, desc="Subtracting bias from darks"), start=1
    ):
        temp = dark['temperature']
        exp = dark['exposure']
        if temp is None:
            continue
        file_path = dark['original_path']

        # Open the raw FITS and read data
        raw_data = load_fits_scaled_12bit(file_path)

        # Find the nearest bias map and subtract
        bias_map = get_nearest_bias_map(temp, bias_map_by_temp)
        corrected_data = raw_data - bias_map

        # Construct FITS HDUs
        hdr_primary = fits.Header()
        hdr_primary['TEMP'] = temp
        hdr_primary['EXPTIME'] = exp
        hdr_primary['BUNIT'] = 'ADU'
        hdr_primary['COMMENT'] = "Bias-corrected dark frame"

        primary_hdu = fits.PrimaryHDU(corrected_data, header=hdr_primary)

        hdr_bias = fits.Header()
        hdr_bias['COMMENT'] = "Bias map used for correction."
        hdr_bias['T_NEAR'] = temp
        bias_hdu = fits.ImageHDU(data=bias_map, header=hdr_bias, name='BIAS_MAP')

        # Write multi-extension FITS
        hdulist = fits.HDUList([primary_hdu, bias_hdu])
        out_name = f"dark_corrected_{idx:04d}.fits"
        out_path = os.path.join(output_dir, out_name)
        hdulist.writeto(out_path, overwrite=True)

        corrected_dark_frames.append({
            'original_path': file_path,
            'corrected_path': out_path,
            'temperature': temp,
            'exposure': exp,
            'corrected_data': corrected_data,
            'bias_map': bias_map
        })

    print(f"Bias-corrected darks saved in: {output_dir}")
    return corrected_dark_frames

def subtract_bias_grouped_by_exposure(
    grouped_darks: dict,
    bias_map_by_temp: dict,
    output_dir: str
    ) -> dict:
    """
    Resta el bias a cada dark agrupado por exposure time. Guarda los resultados
    en subcarpetas por exposure y devuelve un diccionario con los resultados agrupados.

    :param grouped_darks: Diccionario exposure_time -> lista de darks con metadata
    :type grouped_darks: dict[float, list]
    :param bias_map_by_temp: Diccionario temperatura -> bias map
    :type bias_map_by_temp: dict
    :param output_dir: Ruta base donde guardar los archivos FITS corregidos
    :type output_dir: str
    :return: Diccionario exposure_time -> lista de diccionarios con datos corregidos
    :rtype: dict[float, list]
    """
    corrected_by_exposure = {}

    for exposure_time, dark_list in grouped_darks.items():
        exposure_str = f"{exposure_time:.2f}".replace('.', 'p')
        subdir = os.path.join(output_dir, f"exp_{exposure_str}")
        os.makedirs(subdir, exist_ok=True)

        corrected_list = []

        for idx, dark in enumerate(
            tqdm(dark_list, desc=f"Exp {exposure_time}s - Subtracting Bias"), start=1
        ):
            temp = dark['temperature']
            if temp is None:
                continue

            file_path = dark['original_path']
            raw_data = load_fits_scaled_12bit(file_path)

            bias_map = get_nearest_bias_map(temp, bias_map_by_temp)
            corrected_data = raw_data - bias_map

            # Headers
            hdr_primary = fits.Header()
            hdr_primary['TEMP'] = temp
            hdr_primary['EXPTIME'] = exposure_time
            hdr_primary['BUNIT'] = 'ADU'
            hdr_primary['COMMENT'] = "Bias-corrected dark frame"

            primary_hdu = fits.PrimaryHDU(corrected_data, header=hdr_primary)

            hdr_bias = fits.Header()
            hdr_bias['COMMENT'] = "Bias map used for correction."
            hdr_bias['T_NEAR'] = temp
            bias_hdu = fits.ImageHDU(data=bias_map, header=hdr_bias, name='BIAS_MAP')

            hdulist = fits.HDUList([primary_hdu, bias_hdu])
            out_name = f"dark_corrected_{idx:04d}.fits"
            out_path = os.path.join(subdir, out_name)
            hdulist.writeto(out_path, overwrite=True)

            corrected_list.append({
                'original_path': file_path,
                'corrected_path': out_path,
                'temperature': temp,
                'exposure': exposure_time,
                'corrected_data': corrected_data,
                'bias_map': bias_map
            })

        corrected_by_exposure[exposure_time] = corrected_list
        print(f"Guardados {len(corrected_list)} darks corregidos en: {subdir}")

    return corrected_by_exposure
