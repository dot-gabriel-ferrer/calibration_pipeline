import numpy as np
from astropy.io import fits
import pytest

from process_index import _make_mean_master


def _write(path, value):
    fits.writeto(path, np.full((2, 2), value, dtype=np.float32), overwrite=True)


def test_make_mean_master_ignores_invalid_metadata(tmp_path):
    f1 = tmp_path / "f1.fits"
    f2 = tmp_path / "f2.fits"
    _write(f1, 1)
    _write(f2, 2)

    master, hdr = _make_mean_master([str(f1), str(f2)], temps=[10.0, float('inf')], exps=[float('nan'), 1.0])

    assert np.allclose(master, np.full((2, 2), 1.5))
    assert hdr["NSOURCE"] == 2
    assert hdr["TMAX"] == 10.0
    assert "EMAX" in hdr and hdr["EMAX"] == 1.0
