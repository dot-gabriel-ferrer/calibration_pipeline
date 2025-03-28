# utils_scaling.py

import numpy as np
from astropy.io import fits

SCALE_FACTOR = 16.0

def load_fits_scaled_12bit(path):
    """Load FITS file and rescale from 16-bit to 12-bit equivalent."""
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32) / SCALE_FACTOR
    return data
