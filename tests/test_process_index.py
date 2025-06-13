import numpy as np
from astropy.io import fits
import pytest

import pandas as pd
from pathlib import Path

from process_index import _make_mean_master, master_bias_by_temp, master_dark_flat


def _write(path, value):
    fits.writeto(path, np.full((2, 2), value, dtype=np.float32), overwrite=True)


def _make_fits(path, value, temp=None, exp=1.0):
    hdu = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    if temp is not None:
        hdu.header["TEMP"] = temp
        hdu.header["TEMPERATURE"] = temp
    hdu.header["EXPTIME"] = exp
    hdu.writeto(path, overwrite=True)


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


def test_master_dark_flat_bias_correction(tmp_path):
    # setup directory structure matching process_index expectations
    import tempfile

    dataset_root = Path(tempfile.mkdtemp(prefix="data"))
    bias_dir = dataset_root / "run1" / "bias" / "fits"
    dark_dir = dataset_root / "run1" / "dark" / "fits"
    bias_dir.mkdir(parents=True)
    dark_dir.mkdir(parents=True)

    out_bias = dataset_root / "out_bias"
    out_dark = dataset_root / "out_dark"
    out_flat = dataset_root / "out_flat"

    # create bias frames
    b10a = bias_dir / "b_T10a.fits"
    b10b = bias_dir / "b_T10b.fits"
    b20a = bias_dir / "b_T20a.fits"
    b20b = bias_dir / "b_T20b.fits"
    _make_fits(b10a, 1, temp=10.0)
    _make_fits(b10b, 3, temp=10.0)
    _make_fits(b20a, 4, temp=20.0)
    _make_fits(b20b, 6, temp=20.0)

    bias_df = pd.DataFrame(
        {
            "PATH": [str(b10a), str(b10b), str(b20a), str(b20b)],
            "CALTYPE": ["BIAS"] * 4,
            "TEMP": [10.0, 10.0, 20.0, 20.0],
        }
    )

    bias_maps = master_bias_by_temp(bias_df, str(out_bias))

    # create dark frames with constant bias offsets
    d10a = dark_dir / "d_T10a.fits"
    d10b = dark_dir / "d_T10b.fits"
    d20a = dark_dir / "d_T20a.fits"
    d20b = dark_dir / "d_T20b.fits"
    _make_fits(d10a, 100 + 2, temp=10.0, exp=1.0)
    _make_fits(d10b, 150 + 2, temp=10.0, exp=1.0)
    _make_fits(d20a, 200 + 5, temp=20.0, exp=1.0)
    _make_fits(d20b, 300 + 5, temp=20.0, exp=1.0)

    dark_df = pd.DataFrame(
        {
            "PATH": [str(d10a), str(d10b), str(d20a), str(d20b)],
            "CALTYPE": ["DARK"] * 4,
            "TEMP": [10.0, 10.0, 20.0, 20.0],
        }
    )
    flat_df = pd.DataFrame(columns=["PATH", "CALTYPE", "TEMP"])

    dark_maps, _ = master_dark_flat(
        dark_df,
        flat_df,
        str(out_dark),
        str(out_flat),
        bias_maps=bias_maps,
    )

    assert np.allclose(dark_maps[(10.0, 1.0)], np.full((2, 2), 125.0))
    assert np.allclose(dark_maps[(20.0, 1.0)], np.full((2, 2), 250.0))
