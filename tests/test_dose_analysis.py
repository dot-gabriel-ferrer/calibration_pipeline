import os
import numpy as np
import pandas as pd
from astropy.io import fits

from dose_analysis import (
    _dose_from_path,
    _group_paths,
    _make_master,
    _compute_photometric_precision,
    _plot_photometric_precision,
    _save_plot,
    _fit_dose_response,
    _compare_stage_differences,
    _pixel_precision_analysis,
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

    def fake_errorbar(self, x, y, yerr=None, fmt=None, label=None, **k):
        labels.append(label)

    monkeypatch.setattr('matplotlib.axes.Axes.errorbar', fake_errorbar)
    monkeypatch.setattr('matplotlib.axes.Axes.fill_between', lambda *a, **k: None)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig', lambda *a, **k: None)

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
    assert "MAG_ERR_STD" in df.columns


def test_plot_photometric_precision(monkeypatch, tmp_path):
    df = pd.DataFrame({
        "DOSE": [1.0, 2.0],
        "MAG_ERR": [0.1, 0.2],
        "MAG_ERR_STD": [0.01, 0.02],
    })

    yerrs = []

    def fake_errorbar(self, x, y, yerr=None, fmt=None, **k):
        yerrs.append(yerr)

    monkeypatch.setattr("matplotlib.axes.Axes.errorbar", fake_errorbar)
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda *a, **k: None)

    _plot_photometric_precision(df, tmp_path)

    assert len(yerrs) == 1
    assert np.allclose(yerrs[0], df["MAG_ERR_STD"]) 


def test_pixel_precision_analysis_generates_maps(tmp_path):
    bias = tmp_path / 'b.fits'
    dark = tmp_path / 'd.fits'
    _make_fits(bias, 1000)
    _make_fits(dark, 10)

    groups = {
        ('during', 'BIAS', 1.0, None): [str(bias)],
        ('during', 'DARK', 1.0, None): [str(dark)],
    }

    out_dir = tmp_path / 'out'
    stats = _pixel_precision_analysis(groups, str(out_dir))

    assert (out_dir / 'mag_err_1kR.png').is_file()
    assert (out_dir / 'adu_err16_1kR.png').is_file()
    assert (out_dir / 'adu_err12_1kR.png').is_file()
    assert (out_dir / 'mag_err_vs_dose.png').is_file()
    assert (out_dir / 'adu_err_vs_dose.png').is_file()
    assert set(stats.columns) == {"DOSE", "MAG_MEAN", "MAG_STD", "ADU_MEAN", "ADU_STD"}
    assert len(stats) == 1


def test_fit_dose_response_outputs(tmp_path, monkeypatch):
    summary = pd.DataFrame([
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.1},
        {"STAGE": "during", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 0.1},
        {"STAGE": "during", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 0.1},
        {"STAGE": "during", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 2.0, "MEAN": 20.0, "STD": 0.1},
        {"STAGE": "during", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 2.0, "MEAN": 24.0, "STD": 0.1},
    ])

    saved = []

    def fake_savefig(self, path, *a, **k):
        saved.append(os.path.basename(path))

    monkeypatch.setattr('matplotlib.figure.Figure.savefig', fake_savefig)

    _fit_dose_response(summary, tmp_path)

    assert (tmp_path / "dose_model.csv").exists()
    assert sorted(saved) == [
        "dose_model_bias.png",
        "dose_model_dark_E1p0s.png",
        "dose_model_dark_E2p0s.png",
    ]


def test_compare_stage_differences_generates_heatmaps(tmp_path):
    master_dir = tmp_path / "masters"
    out_dir = tmp_path / "out"
    master_dir.mkdir()

    # create masters for bias
    _make_fits(master_dir / "master_bias_pre_D1kR_E1.0.fits", 1)
    _make_fits(master_dir / "master_bias_during_D1kR_E1.0.fits", 2)
    _make_fits(master_dir / "master_bias_during_D5kR_E1.0.fits", 5)
    _make_fits(master_dir / "master_bias_post_D5kR_E1.0.fits", 4)

    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.0},
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.0},
        {"STAGE": "during", "CALTYPE": "BIAS", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 5.0, "STD": 0.0},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 4.0, "STD": 0.0},
    ])

    _compare_stage_differences(summary, str(master_dir), str(out_dir))

    assert (out_dir / "stage_differences.csv").is_file()
    assert (out_dir / "bias_first_vs_pre.png").is_file()
    assert (out_dir / "bias_post_vs_last.png").is_file()

