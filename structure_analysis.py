#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo COMPLETO y EJECUTABLE para:
 - Cargar observaciones usando ObservationManager
 - Filtrar darks largos y bias (darks cortos)
 - Revisar bias sospechosos y generar mapas de bias
 - Cargar flats (sin reducir o reducidos, según configuración)
 - Calcular estadísticas de variación media de píxeles con el tiempo y la temperatura
 - Detectar estructuras nuevas que aparezcan en los flats
 - Generar plots de resultados

IMPORTANTE:
 - Asume que tienes instalado astropy, numpy, matplotlib, tqdm.
 - Asume que tienes disponible la clase 'ObservationManager' en el path:
       from observation_manager.observation_manager import ObservationManager
   (modifica el import según tu estructura de paquetes).
 - Ajusta rutas, tolerancias de temperatura, umbrales, etc. a tus necesidades reales.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from astropy.io import fits

# -----------------------------------------------------------------------------
# IMPORTAR ObservationManager
# -----------------------------------------------------------------------------
# Modifica este import si tu clase se encuentra en otro lugar o módulo:
try:
    from observation_manager.observation_manager import ObservationManager
except ImportError:
    # Si tuvieras el package local, podrías usar:
    # from .observation_manager.observation_manager import ObservationManager
    print("No se pudo importar ObservationManager. Asegúrate de que está instalado o en el PYTHONPATH.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1) CARGA DE OBSERVACIONES: Filtrar darks largos y bias (darks cortos)
# -----------------------------------------------------------------------------
def load_observations(basepath: str):
    """
    Carga y organiza ficheros usando ObservationManager,
    y filtra para obtener:
      - darks largos (long_darks)
      - bias (short_darks)

    Ajusta los parámetros exp_min / exp_max según tu criterio.
    """
    manager = ObservationManager(base_path=basepath)
    manager.load_and_organize()

    # Filtra darks largos (ejemplo: exp_min=0.1)
    long_darks = manager.filter_files(
        category='CALIBRATION',
        subcat='darks',
        exp_min=0.1,
        ext_filter='fits'
    )

    # Filtra bias frames (dark muy corto), exp_max=0.1 por ejemplo
    short_darks = manager.filter_files(
        category='CALIBRATION',
        subcat='bias',
        exp_max=0.1,
        ext_filter='fits'
    )

    return manager, long_darks, short_darks


# -----------------------------------------------------------------------------
# 2) REVISIÓN DE BIAS Y GENERACIÓN DE MAPAS DE BIAS
# -----------------------------------------------------------------------------
def check_bias_frames(short_darks, zero_fraction_threshold=0.1, log_path='suspicious_bias.log'):
    """
    Revisa la lista de bias para encontrar aquellos con una fracción alta
    de píxeles en cero, considerándolos sospechosos. Genera un log.

    Devuelve (lista_bias_buenos, lista_bias_sospechosos).
    """
    good_bias = []
    suspicious_bias = []

    with open(log_path, 'w') as log_file:
        log_file.write("Bias con grandes zonas de pixeles en 0:\n")

        for dark in tqdm(short_darks, desc="Revisando Bias"):
            file_path = dark['original_path']
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
            
            zero_fraction = np.mean(data == 0)

            if zero_fraction > zero_fraction_threshold:
                suspicious_bias.append(dark)
                log_file.write(f"{file_path} -> Fracción de píxeles en 0: {zero_fraction:.2%}\n")
            else:
                good_bias.append(dark)

    return good_bias, suspicious_bias


def generate_bias_maps(short_darks: list) -> dict:
    """
    Genera un diccionario {temp: bias_map_2D}, promediando todos los frames
    de la misma temperatura. Temp se obtiene de short_darks[i]['temperature'].
    """
    bias_dict = {}
    
    for dark in tqdm(short_darks, desc="Generando mapas de Bias"):
        temp = dark['temperature']
        if temp is None:
            continue
        file_path = dark['original_path']
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)

        bias_dict.setdefault(temp, []).append(data)

    bias_map_by_temp = {}
    for temp, frames in bias_dict.items():
        stack = np.stack(frames, axis=0)
        bias_map = np.mean(stack, axis=0)
        bias_map_by_temp[temp] = bias_map

    return bias_map_by_temp


# -----------------------------------------------------------------------------
# 3) GENERACIÓN DE DARK MAPS (OPCIONAL) PARA SUSTRAER DE LOS FLATS
# -----------------------------------------------------------------------------
def generate_dark_maps(long_darks: list, temp_tolerance=1.0, exp_tolerance=0.1) -> dict:
    """
    Genera un diccionario con llaves (temp, exp_time) y valores un dark map 2D promedio.
    Cada dark en 'long_darks' se agrupa según su temperatura 'temp' y su 'exposure_time' 
    discretizadas dentro de tolerancias.

    Para simplificar, se hace un 'redondeo' a un paso que consideres (por ejemplo, .1 en temperatura, .1 en exp_time).
    Ajusta a tus necesidades reales.
    """
    # Agruparemos (temp_round, exp_round) -> lista de arrays
    group_dict = {}
    for dark in tqdm(long_darks, desc="Generando mapas de Dark"):
        temp = dark.get('temperature')
        exp = dark.get('exposure_time')
        if temp is None or exp is None:
            continue

        # Redondeo (ajusta si lo ves necesario)
        temp_rounded = round(temp, 1)
        exp_rounded = round(exp, 1)

        file_path = dark['original_path']
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)

        group_dict.setdefault((temp_rounded, exp_rounded), []).append(data)

    # Promedio por cada (temp_rounded, exp_rounded)
    dark_map_by_temp_exp = {}
    for key, frames in group_dict.items():
        stack = np.stack(frames, axis=0)
        dark_map = np.mean(stack, axis=0)
        dark_map_by_temp_exp[key] = dark_map

    return dark_map_by_temp_exp


# -----------------------------------------------------------------------------
# 4) CARGAR FLATS Y (OPCIONAL) SUSTRAER BIAS/DARK
# -----------------------------------------------------------------------------
def find_closest_dark(temp, exp_time, dark_map_dict, temp_tol=1.0, exp_tol=0.1):
    """
    Busca en dark_map_dict (cuyas claves son (temp_rounded, exp_rounded))
    el mapa más cercano a (temp, exp_time) dentro de las tolerancias dadas.
    Devuelve el array 2D del dark, o None si no encuentra nada adecuado.
    """
    if not dark_map_dict:
        return None
    
    # Ordenar por proximidad en temp y exp_time
    candidates = []
    for (t_key, e_key), dark_data in dark_map_dict.items():
        diff_temp = abs(t_key - round(temp, 1))
        diff_exp = abs(e_key - round(exp_time, 1))
        if diff_temp <= temp_tol and diff_exp <= exp_tol:
            dist = diff_temp + diff_exp
            candidates.append((dist, dark_data))
    
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])  # menor distancia primero
    return candidates[0][1]


def load_flats(manager, do_reduction=False, bias_map_by_temp=None, dark_map_by_temp_exp=None):
    """
    Carga flats desde el manager. Por defecto no hace reducción (do_reduction=False).
    Si do_reduction=True, se resta bias y dark (si existen).

    bias_map_by_temp: dict {temp: bias_2D}
    dark_map_by_temp_exp: dict {(temp_round, exp_round): dark_2D}
    """
    flats = manager.filter_files(
        category='CALIBRATION',
        subcat='flats',
        ext_filter='fits'
    )

    flats_data = []
    for flat_info in tqdm(flats, desc="Cargando Flats"):
        file_path = flat_info['original_path']
        temp = flat_info.get('temperature', None)
        exp_time = flat_info.get('exposure_time', None)

        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)

        if do_reduction:
            # Sustraer bias si hay para esa temperatura
            if (bias_map_by_temp is not None) and (temp in bias_map_by_temp):
                data -= bias_map_by_temp[temp]

            # Sustraer dark si corresponde y si tenemos info
            if (dark_map_by_temp_exp is not None) and (temp is not None) and (exp_time is not None):
                dark_data = find_closest_dark(
                    temp=temp, 
                    exp_time=exp_time,
                    dark_map_dict=dark_map_by_temp_exp,
                    temp_tol=1.0,
                    exp_tol=0.1
                )
                if dark_data is not None:
                    data -= dark_data

        # Guardamos los datos
        flat_info_copy = flat_info.copy()
        flat_info_copy['data'] = data
        flats_data.append(flat_info_copy)

    return flats_data


# -----------------------------------------------------------------------------
# 5) ANÁLISIS: Variación de píxeles en el tiempo (misma T) y vs temperatura
# -----------------------------------------------------------------------------
def extract_datetime(obs_dict):
    """
    Extrae un objeto datetime desde la info del manager.
    Ajusta la key donde se guarda la fecha/hora en tu manager (p.ej. 'date_obs', 'datetime_obs', etc.).
    Si no existe, retorna None.
    """
    # Ejemplo: supongamos que 'date_obs' es un string 'YYYY-MM-DDTHH:MM:SS'
    date_str = obs_dict.get('date_obs', None)
    if not date_str:
        return None
    try:
        # Ajusta el formato si no es así
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except:
        return None


def plot_variation_time(flats_data):
    """
    Para cada temperatura, traza la variación de la media de pixeles en función del tiempo.
    """
    # Agrupamos flats por temperatura
    from collections import defaultdict
    flats_by_temp = defaultdict(list)

    for flat in flats_data:
        T = flat.get('temperature', None)
        dt = extract_datetime(flat)
        data = flat['data']
        mean_val = float(np.mean(data))
        if T is not None and dt is not None:
            flats_by_temp[T].append((dt, mean_val))

    # Para cada temperatura, ordenamos en el tiempo y hacemos un plot
    for T, measurements in flats_by_temp.items():
        if len(measurements) < 2:
            continue
        # Ordenar por fecha
        measurements.sort(key=lambda x: x[0])
        times = [m[0] for m in measurements]
        vals = [m[1] for m in measurements]

        # Convertir tiempos a matplotlib
        import matplotlib.dates as mdates
        times_num = mdates.date2num(times)

        plt.figure()
        plt.plot_date(times_num, vals, marker='o', linestyle='-')
        plt.title(f"Variación media de píxeles vs Tiempo (T={T:.1f} C)")
        plt.xlabel("Tiempo")
        plt.ylabel("Media de píxeles (ADU)")
        plt.gcf().autofmt_xdate()
        plt.show()


def plot_variation_temp(flats_data):
    """
    Traza la variación de la media de píxeles en función de la temperatura, 
    ignorando el tiempo.
    """
    temps = []
    means = []

    for flat in flats_data:
        T = flat.get('temperature')
        data = flat['data']
        if T is not None:
            temps.append(T)
            means.append(float(np.mean(data)))

    plt.figure()
    plt.scatter(temps, means)
    plt.title("Variación media de píxeles vs Temperatura")
    plt.xlabel("Temperatura (C)")
    plt.ylabel("Media de píxeles (ADU)")
    plt.show()


# -----------------------------------------------------------------------------
# 6) DETECCIÓN DE ESTRUCTURAS NUEVAS EN LOS FLATS
# -----------------------------------------------------------------------------
def detect_new_structures(flats_data, outlier_sigma=5.0):
    """
    Busca estructuras nuevas comparando cada flat con el primero de su temperatura.
    Si la diferencia local excede outlier_sigma * std_diff, lo marcamos como outlier.

    Retorna una lista con info de estructuras detectadas.
    """
    # Para cada temperatura, usaremos el primer flat como referencia
    reference_dict = {}
    detections = []

    for flat in tqdm(flats_data, desc="Buscando estructuras nuevas"):
        T = flat.get('temperature')
        dt = extract_datetime(flat)
        data = flat['data']

        if T not in reference_dict:
            # Este flat será la referencia para su temperatura
            reference_dict[T] = data
            continue

        ref_data = reference_dict[T]
        if ref_data.shape != data.shape:
            # Si no coincide tamaño, saltamos
            continue

        diff = data - ref_data
        std_diff = np.std(diff)

        # Máscara de outliers
        mask = np.abs(diff) > (outlier_sigma * std_diff)
        num_outliers = np.sum(mask)

        if num_outliers > 0:
            # Podríamos etiquetar y encontrar regiones conectadas. Aquí, solo guardamos el total.
            detection_info = {
                "datetime": dt,
                "temperature": T,
                "num_outlier_pixels": int(num_outliers),
                "std_diff": float(std_diff),
            }
            detections.append(detection_info)

    return detections


def visualize_first_detection(detections, flats_data):
    """
    Si hay detecciones, muestra un imshow de la primera detección
    comparada con su referencia.
    """
    if not detections:
        print("No se han detectado estructuras nuevas.")
        return

    # Ordenar detecciones por tiempo
    detections.sort(key=lambda x: x["datetime"] or datetime.max)
    first = detections[0]

    Tfd = first["temperature"]
    dtfd = first["datetime"]

    # Localizar la data del flat 'dtfd, Tfd' y la de su referencia
    target_flat = None
    ref_flat = None
    for flat in flats_data:
        T = flat.get('temperature')
        dt = extract_datetime(flat)
        if T == Tfd and dt == dtfd:
            target_flat = flat['data']
        if T == Tfd and dt is not None:
            # El primero que entró en reference_dict en detect_new_structures
            # fue el primer flat encontrado. Para reproducirlo, asumimos
            # que ese es el flat con la fecha/hora mínima
            # (puedes llevar un dict aparte para mayor exactitud).
            # Aquí haremos algo sencillo: localizamos el flat con esa T 
            # y la hora más antigua.
            if (ref_flat is None) or (dt < extract_datetime(ref_flat)):
                ref_flat = flat

    if (target_flat is None) or (ref_flat is None):
        print("No se pudo localizar el flat detectado o su referencia para visualización.")
        return

    diff_data = target_flat - ref_flat['data']
    
    plt.figure()
    plt.imshow(diff_data, origin='lower')
    plt.title(f"Diferencia vs Referencia\nT={Tfd:.1f}C, {dtfd}")
    plt.colorbar(label="Diferencia (ADU)")
    plt.show()


# -----------------------------------------------------------------------------
# MAIN: EJECUCIÓN COMPLETA
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ejemplo de script para cargar, generar bias, cargar flats y analizar degradación."
    )
    parser.add_argument("basepath", help="Directorio base con los ficheros FITS.")
    parser.add_argument("--reduce", action="store_true", 
                        help="Si se especifica, se hará la reducción de flats (resta de bias/dark).")
    args = parser.parse_args()

    # 1) Carga de observaciones
    manager, long_darks, short_darks = load_observations(args.basepath)
    print(f"Se cargaron {len(long_darks)} darks largos y {len(short_darks)} bias (darks cortos).")

    # 2) Revisar bias y generar mapas
    good_bias, suspicious_bias = check_bias_frames(short_darks)
    print(f"Bias buenos: {len(good_bias)} | Sospechosos: {len(suspicious_bias)}")
    bias_map_by_temp = generate_bias_maps(good_bias)

    # 3) Generar mapas de dark (opcional si quieres restarlos)
    dark_map_by_temp_exp = generate_dark_maps(long_darks)

    # 4) Cargar flats (opcionalmente reducidos)
    flats_data = load_flats(
        manager=manager,
        do_reduction=args.reduce, 
        bias_map_by_temp=bias_map_by_temp,
        dark_map_by_temp_exp=dark_map_by_temp_exp
    )
    print(f"Se cargaron {len(flats_data)} flats. Reducción={args.reduce}")

    # 5a) Plot variación media de píxeles con el tiempo (para cada T)
    plot_variation_time(flats_data)

    # 5b) Plot variación media de píxeles vs temperatura (ignorando tiempo)
    plot_variation_temp(flats_data)

    # 6) Detección de estructuras nuevas con el tiempo
    detections = detect_new_structures(flats_data, outlier_sigma=5.0)
    print(f"Se han detectado {len(detections)} flats con estructuras potencialmente nuevas.")
    for d in detections:
        dtstr = d["datetime"].isoformat() if d["datetime"] else "N/A"
        print(f" - T={d['temperature']}C, t={dtstr}, "
              f"Outliers={d['num_outlier_pixels']}, std_diff={d['std_diff']:.2f}")

    # Visualizamos la primera detección (si existe)
    visualize_first_detection(detections, flats_data)


if __name__ == "__main__":
    main()
