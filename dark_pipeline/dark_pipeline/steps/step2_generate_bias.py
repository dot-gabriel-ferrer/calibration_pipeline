# Author: Elías Gabriel Ferrer Jorge

"""
Step 2: Generate bias maps by averaging the short dark (bias) frames for each temperature.
"""

import numpy as np
from astropy.io import fits
from tqdm import tqdm
from .utils.utils_scaling import load_fits_scaled_12bit

def check_bias_frames(short_darks, zero_fraction_threshold=0.1, log_path='suspicious_bias.log'):
    """
    Revisa la lista de bias para encontrar aquellos que tienen grandes zonas con píxeles en 0.
    Se entiende que 'grandes zonas' se pueden aproximar revisando la fracción de píxeles en 0
    respecto al total de la imagen.

    :param short_darks: Lista de diccionarios que describen los bias frames, con campos como 'original_path' y 'temperature'.
    :type short_darks: list
    :param zero_fraction_threshold: Umbral de fracción de píxeles en cero para marcar el bias como sospechoso.
    :type zero_fraction_threshold: float
    :param log_path: Ruta del fichero de log donde se guardarán los bias sospechosos.
    :type log_path: str
    :return: (lista de bias buenos, lista de bias sospechosos)
    :rtype: (list, list)
    """
    good_bias = []
    suspicious_bias = []

    with open(log_path, 'w') as log_file:
        log_file.write("Bias con grandes zonas de pixeles en 0:\n")

        for dark in tqdm(short_darks, desc="Checking Frames"):
            file_path = dark['original_path']
            
            # Leemos la imagen
            data = load_fits_scaled_12bit(file_path)
            
            # Calculamos la fracción de píxeles que están a cero
            zero_fraction = np.mean(data == 0)

            if zero_fraction > zero_fraction_threshold:
                # Se considera sospechoso
                suspicious_bias.append(dark)
                # Lo escribimos en el fichero de log
                log_file.write(f"{file_path} -> Fracción de píxeles en 0: {zero_fraction:.2%}\n")
            else:
                # Se considera apto
                good_bias.append(dark)

    return good_bias, suspicious_bias


def generate_bias_maps(short_darks: list) -> dict:
    """
    Crea un diccionario de mapas de bias, indexado por la temperatura,
    promediando todos los frames que correspondan a cada temperatura.

    :param short_darks: Lista de diccionarios que describen los bias frames, 
                        cada uno con campos como 'original_path', 'temperature', etc.
    :type short_darks: list
    :return: Diccionario que mapea float(temperature) al bias map (2D numpy array).
    :rtype: dict
    """
    import numpy as np
    from astropy.io import fits
    from tqdm import tqdm

    bias_dict = {}
    
    # Recopilamos frames por temperatura
    for dark in tqdm(short_darks, desc="Generating Bias Maps"):
        temp = dark['temperature']
        if temp is None:
            continue

        file_path = dark['original_path']
        data = load_fits_scaled_12bit(file_path)
        bias_dict.setdefault(temp, []).append(data)

    # Calculamos el promedio por temperatura
    bias_map_by_temp = {}
    for temp, frames in bias_dict.items():
        stack = np.stack(frames, axis=0)
        bias_map = np.mean(stack, axis=0)
        bias_map_by_temp[temp] = bias_map

    return bias_map_by_temp
