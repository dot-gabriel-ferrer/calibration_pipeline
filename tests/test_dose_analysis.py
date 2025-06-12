import numpy as np
import pandas as pd
from astropy.io import fits

from dose_analysis import _dose_from_path, _group_paths, _make_master


def _make_fits(path, value, temp=10.0, exp=1.0):
    hdu = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    hdu.header['TEMP'] = temp
    hdu.header['EXPTIME'] = exp
    hdu.writeto(path, overwrite=True)


def test_dose_parsing():
    assert _dose_from_path('a/10kRads/file.fits') == 10.0
    assert _dose_from_path('no_dose/file.fits') is None


def test_group_and_master(tmp_path):
    f1 = tmp_path / 'Bias_1kRads_exp1sframe0.fits'
    f2 = tmp_path / 'Bias_1kRads_exp1sframe1.fits'
    _make_fits(f1, 1, temp=10.0, exp=1.0)
    _make_fits(f2, 3, temp=12.0, exp=1.0)

    df = pd.DataFrame({
        'PATH': [str(f1), str(f2)],
        'CALTYPE': ['BIAS', 'BIAS'],
        'STAGE': ['during', 'during'],
        'VACUUM': ['air', 'air'],
        'TEMP': [10.0, 12.0],
        'ZEROFRACTION': [0.0, 0.0],
        'BADFITS': [False, False],
    })

    groups = _group_paths(df)
    key = ('during', 'BIAS', 1.0, 1.0)
    assert key in groups and len(groups[key]) == 2

    master, hdr = _make_master(groups[key])
    assert hdr['NSOURCE'] == 2
    assert 'T_MEAN' in hdr and abs(hdr['T_MEAN'] - 11.0) < 1e-6
