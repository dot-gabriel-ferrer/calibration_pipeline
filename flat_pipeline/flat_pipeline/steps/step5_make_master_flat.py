# Author: Elías Gabriel Ferrer Jorge
#
# Step 5: Generate final Master Flats from the vignetting-corrected flats (or fallback to normalized if needed).
# --------------------------------------------------------------------------

import os
import numpy as np
from astropy.io import fits
from collections import defaultdict
from tqdm import tqdm


def group_flats_for_master(flat_entries, grouping=('FILTER', 'temperature', 'exposure')):
    """
    Agrupa la lista de flats en función de determinadas claves (por defecto: FILTER, temperature, exposure).
    Devuelve un dict: { (filtro, temp, exp): [lista_de_flats] }
    """
    groups = defaultdict(list)
    for entry in flat_entries:
        # Construimos la key de agrupación
        group_key = []
        for key in grouping:
            group_key.append(entry.get(key, "UNKNOWN"))
        group_key = tuple(group_key)
        groups[group_key].append(entry)

    return dict(groups)


def combine_master_flat(file_entries, method='median'):
    """
    Carga cada flat de 'file_entries', los apila y los combina (median o mean).
    Devuelve (combined_data, combined_header).
    """
    data_stack = []
    used_paths = []
    temps = []
    exps = []
    filters_used = []

    for entry in file_entries:
        path = entry['original_path']
        used_paths.append(os.path.basename(path))
        temps.append(entry.get('temperature', -999))
        exps.append(entry.get('exposure', -1))
        filters_used.append(str(entry.get('FILTER', 'UNKNOWN')))

        with fits.open(path) as hdul:
            data_stack.append(hdul[0].data.astype(np.float32))

    data_stack = np.stack(data_stack, axis=0)

    if method == 'median':
        combined_data = np.median(data_stack, axis=0)
    elif method == 'mean':
        combined_data = np.mean(data_stack, axis=0)
    else:
        raise ValueError("Método de combinación no soportado. Usa 'median' o 'mean'.")

    # Construimos un header con metadatos agregados
    combined_header = fits.Header()
    combined_header['FLATPIPE'] = ('master_flat', 'Pipeline step5 result')
    combined_header['N_FILES'] = (len(file_entries), 'Number of combined flats')
    combined_header['COMBMETH'] = (method.upper(), 'Combination method used')

    combined_header['DATAMIN'] = float(np.nanmin(combined_data))
    combined_header['DATAMAX'] = float(np.nanmax(combined_data))
    combined_header['DATAMEAN'] = float(np.nanmean(combined_data))

    # Ejemplo: guardamos nombre de primer y último archivo
    combined_header['SRC1'] = used_paths[0]
    combined_header['SRCLAST'] = used_paths[-1]

    # Documentamos temperatura, exposición y filtro aproximados
    combined_header['T_AVG'] = float(np.mean(temps))
    combined_header['E_AVG'] = float(np.mean(exps))
    combined_header['F_SAMP'] = filters_used[0]

    return combined_data.astype(np.float32), combined_header


def generate_master_flats(flat_entries,
                          output_dir,
                          grouping=('FILTER', 'temperature', 'exposure'),
                          method='median',
                          fallback_entries=None):
    """
    Step 5 principal: genera los master flats a partir de una lista de flats vignetting-corrected.
    Si esa lista está vacía y se ha proporcionado 'fallback_entries' (por ejemplo, flats normalizados),
    usa esos en su lugar.

    :param flat_entries: list[dict], cada dict contiene metadatos y 'original_path' a los flats vignetting-corrected.
    :param output_dir: Carpeta donde se guardarán los master flats.
    :param grouping: Claves por las que agrupar (por defecto: FILTER, temperature, exposure).
    :param method: 'median' o 'mean'.
    :param fallback_entries: list[dict], lista de flats (ej. normalizados) a usar si flat_entries está vacío.
    """
    print("\n[Step 5] Generating Master Flats...")
    os.makedirs(output_dir, exist_ok=True)

    if not flat_entries:
        if fallback_entries:
            print("[Step 5] No vignetting-corrected flats. Falling back to normalized.")
            flat_entries = fallback_entries
        else:
            print("[Step 5] No input flats found. Skipping Master Flat generation.")
            return

    groups = group_flats_for_master(flat_entries, grouping=grouping)

    if not groups:
        print("[Step 5] After grouping, no valid flats found. Exiting.")
        return

    for group_key, entries in tqdm(groups.items(), desc="Combining flats", ncols=80):
        combined_data, combined_header = combine_master_flat(entries, method=method)

        fval = str(group_key[0]).replace(" ", "")
        tval = group_key[1]
        eval_ = group_key[2]

        out_name = f"master_flat_{fval}_T{tval}_E{eval_}.fits"
        out_path = os.path.join(output_dir, out_name)

        hdu = fits.PrimaryHDU(data=combined_data, header=combined_header)
        hdul_out = fits.HDUList([hdu])
        hdul_out.writeto(out_path, overwrite=True)

    print(f"[Step 5] Master flats saved to '{output_dir}'\n")
