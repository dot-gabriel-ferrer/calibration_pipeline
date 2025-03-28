def load_corrected_dark_frames(path_to_corrected: str) -> list:
    """
    Carga los .FITS corregidos y devuelve una lista
    de dicts con la misma estructura que produce subtract_bias_from_darks:
    [
      {
        'original_path': ...,
        'corrected_path': ...,
        'temperature': ...,
        'exposure': ...,
        'corrected_data': ...,
        'bias_map': ...
      },
      ...
    ]
    """
    # Por ejemplo:
    import glob
    from astropy.io import fits
    import numpy as np
    import os
    from tqdm import tqdm

    corrected_info = []
    fits_paths = glob.glob(os.path.join(path_to_corrected, "*.fits"))
    for fpath in tqdm(fits_paths, desc="Loading Corrected Darks and Bias Maps"):
        with fits.open(fpath) as hdul:
            # El PrimaryHDU es la data corregida
            corrected_data = hdul[0].data
            temp = hdul[0].header.get("TEMP")
            exp = hdul[0].header.get("EXPTIME")

            # El HDU 'BIAS_MAP' es la bias map usada
            # (puede que quieras manejar que no exista, etc.)
            if len(hdul) > 1 and hdul[1].name == "BIAS_MAP":
                bias_map = hdul[1].data
            else:
                bias_map = None

            corrected_info.append({
                'original_path': None,  # si no tienes esa info, puedes poner None
                'corrected_path': fpath,
                'temperature': temp,
                'exposure': exp,
                'corrected_data': corrected_data,
                'bias_map': bias_map
            })
    return corrected_info
