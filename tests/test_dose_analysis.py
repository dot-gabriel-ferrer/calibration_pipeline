import numpy as np
import pandas as pd
from astropy.io import fits

from dose_analysis import (
    _dose_from_path,
    _group_paths,
    _make_master,
    _compute_photometric_precision,
    _save_plot,
)



def _make_fits(path, value, temp=10.0, exp=1.0):
    hdu = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    hdu.header['TEMP'] = temp
    hdu.header['EXPTIME'] = exp
    hdu.writeto(path, overwrite=True)


def test_dose_parsing():
    assert _dose_from_path('a/10kRads/file.fits') == 10.0
    assert _dose_from_path('no_dose/file.fits') is None


def test_group_and_master(tmp_path):
    f1 = tmp_path / 'Bias_1kRads_E1.0_frame0.fits'
    f2 = tmp_path / 'Bias_1kRads_E1.0_frame1.fits'
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


def test_save_plot_all_stages(monkeypatch, tmp_path):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.2},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.3},
    ])

    labels = []

    def fake_errorbar(x, y, yerr=None, fmt=None, label=None, **k):
        labels.append(label)

    monkeypatch.setattr('matplotlib.pyplot.errorbar', fake_errorbar)
    monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *a, **k: None)

    _save_plot(summary, tmp_path)

    assert sorted(labels) == ['during', 'post', 'pre']


def test_compute_photometric_precision():
    summary = pd.DataFrame([
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 0.0, "MEAN": 1000.0, "STD": 4.0},
        {"STAGE": "during", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 100.0, "STD": 5.0},
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 0.0, "MEAN": 1000.0, "STD": 4.0},
        {"STAGE": "during", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 100.0, "STD": 10.0},
    ])

    df = _compute_photometric_precision(summary)
    assert set(df['DOSE']) == {1.0, 2.0}
    assert df['MAG_ERR'].iloc[0] < df['MAG_ERR'].iloc[1]

