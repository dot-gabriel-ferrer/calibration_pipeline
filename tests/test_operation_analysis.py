import numpy as np
from astropy.io import fits

from operation_analysis import _load_frames


def test_load_frames_numeric_order(tmp_path):
    attempt = tmp_path / "attempt"
    fits_dir = attempt / "fits"
    fits_dir.mkdir(parents=True)

    # create files in non-numeric order
    names = ["f10.fits", "f2.fits", "f1.fits"]
    for name in names:
        fits.writeto(fits_dir / name, np.zeros((1, 1), dtype=np.float32), overwrite=True)

    expected = [
        str(fits_dir / "f1.fits"),
        str(fits_dir / "f2.fits"),
        str(fits_dir / "f10.fits"),
    ]
    assert _load_frames(str(attempt)) == expected
