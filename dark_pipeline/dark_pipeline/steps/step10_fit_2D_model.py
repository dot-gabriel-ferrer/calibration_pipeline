# step10_fit_2D_model.py
# Autor: Elías Gabriel Ferrer Jorge

"""
Modelo 2D optimizado:
DC(T, t_exp) = A * t_exp^gamma * exp(B * (T - t_min))
Evita saturación cargando FITS por bloques y sólo almacenando valores de hot pixels.
"""

from collections import defaultdict
import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import glob
import re

def parse_temp_and_exptime_from_filename(fname):
    pattern = r"dark_mask_t([-\d.]+)_e([\d.]+).fits"
    match = re.search(pattern, fname)
    if match:
        temp = float(match.group(1))
        exp = float(match.group(2))
        return temp, exp
    return None, None

def fit_model_2D_pixel(temps, exps, dc_values, t_min):
    temps = np.array(temps)
    exps = np.array(exps)
    dc_values = np.array(dc_values)

    mask = (dc_values > 0) & (exps > 0)
    if np.count_nonzero(mask) < 3:
        return 0.0, 0.0, 0.0

    T_fit = temps[mask]
    E_fit = exps[mask]
    DC_fit = dc_values[mask]

    log_DC = np.log(DC_fit)
    log_E = np.log(E_fit)
    delta_T = T_fit - t_min

    A_matrix = np.vstack([log_E, delta_T, np.ones(len(log_DC))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, log_DC, rcond=None)
    gamma, B, log_A = coeffs
    A = np.exp(log_A)
    return A, B, gamma

def fit_dark_model_2D(masks_root, hot_pixel_mask_path, output_dir, max_files=100):
    print("[1/4] Cargando máscara de hot pixels...")
    with fits.open(hot_pixel_mask_path) as hdul:
        hot_pixels = hdul[0].data.astype(bool)
    shape = hot_pixels.shape
    hot_indices = np.argwhere(hot_pixels)

    print(f"[✓] Se detectaron {len(hot_indices)} hot pixels.")

    print("[2/4] Buscando FITS disponibles...")
    all_fits = sorted(glob.glob(os.path.join(masks_root, "exp_*/dark_mask_t*_e*.fits")))

    print("[✓] Agrupando FITS por temperatura y exposición...")
    buckets = defaultdict(list)
    for f in all_fits:
        temp, exp = parse_temp_and_exptime_from_filename(os.path.basename(f))
        if temp is not None and exp is not None:
            key = (round(temp, 1), round(exp, 1))
            buckets[key].append(f)

    selected = []
    n_per_bucket = max(1, max_files // len(buckets)) if max_files else None
    for group in buckets.values():
        selected.extend(sorted(group)[:n_per_bucket])
    selected = selected[:max_files] if max_files else selected

    print(f"[✓] Se usarán {len(selected)} FITS tras muestreo equilibrado.")

    print("[3/4] Cargando valores DC para todos los hot pixels...")
    hot_pixel_series = defaultdict(list)
    temps_all = []

    for fpath in tqdm(selected, desc="Cargando bloques", leave=False):
        temp, exp = parse_temp_and_exptime_from_filename(os.path.basename(fpath))
        if temp is None or exp is None:
            continue
        with fits.open(fpath) as hdul:
            data = hdul[0].data.astype(np.float32)
        temps_all.append(temp)

        for y, x in hot_indices:
            dc_val = data[y, x]
            hot_pixel_series[(y, x)].append((temp, exp, dc_val))

    t_min = min(temps_all)
    print(f"[✓] Usando t_min = {t_min:.2f} para normalización.")

    print("[4/4] Ajustando modelo por píxel...")
    A_map = np.zeros(shape, dtype=np.float32)
    B_map = np.zeros(shape, dtype=np.float32)
    GAMMA_map = np.zeros(shape, dtype=np.float32)

    for (y, x), series in tqdm(hot_pixel_series.items(), desc="Ajustando modelo", leave=False):
        temps, exps, vals = zip(*series)
        A, B, gamma = fit_model_2D_pixel(temps, exps, vals, t_min)
        A_map[y, x] = A
        B_map[y, x] = B
        GAMMA_map[y, x] = gamma

    os.makedirs(output_dir, exist_ok=True)
    fits.PrimaryHDU(A_map).writeto(os.path.join(output_dir, "A_map_2D.fits"), overwrite=True)
    fits.PrimaryHDU(B_map).writeto(os.path.join(output_dir, "B_map_2D.fits"), overwrite=True)
    fits.PrimaryHDU(GAMMA_map).writeto(os.path.join(output_dir, "GAMMA_map_2D.fits"), overwrite=True)
    np.savez(os.path.join(output_dir, "global_params_2d.npz"), t_min=t_min)
    print(f"[✓] Modelo 2D ajustado y guardado en: {output_dir}")

