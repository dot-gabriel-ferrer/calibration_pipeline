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
    _fit_base_level_trend,
    _stage_base_level_diff,
    _compare_stage_differences,
    _pixel_precision_analysis,
    _dynamic_range_analysis,
    _relative_precision_analysis,
    _plot_bias_dark_error,
    _estimate_dose_rate,
    _plot_dose_rate_effect,
)



def _make_fits(path, value, temp=10.0, exp=1.0, ts=None):
    hdu = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    hdu.header['TEMP'] = temp
    hdu.header['EXPTIME'] = exp
    if ts is not None:
        hdu.header['TIMESTAMP'] = ts
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
        'STAGE': ['radiating', 'radiating'],
        'VACUUM': ['air', 'air'],
        'TEMP': [10.0, 12.0],
        'ZEROFRACTION': [0.0, 0.0],
        'BADFITS': [False, False],
    })

    groups = _group_paths(df)
    key = ('radiating', 'BIAS', 1.0, 1.0)
    assert key in groups and len(groups[key]) == 2

    master, hdr = _make_master(groups[key])
    assert hdr['NSOURCE'] == 2
    assert 'T_MEAN' in hdr and abs(hdr['T_MEAN'] - 11.0) < 1e-6


def test_save_plot_all_stages(monkeypatch, tmp_path):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.2},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.3},
    ])

    labels = []

    def fake_errorbar(self, x, y, yerr=None, fmt=None, label=None, **k):
        labels.append(label)

    monkeypatch.setattr('matplotlib.axes.Axes.errorbar', fake_errorbar)
    monkeypatch.setattr('matplotlib.axes.Axes.fill_between', lambda *a, **k: None)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig', lambda *a, **k: None)

    _save_plot(summary, tmp_path)

    assert sorted(labels) == ['post', 'pre', 'radiating']
    assert (tmp_path / 'bias_mean_vs_dose_E1p0s.npz').is_file()


def test_compute_photometric_precision():
    summary = pd.DataFrame([
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 0.0, "MEAN": 1000.0, "STD": 4.0},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 100.0, "STD": 5.0},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 0.0, "MEAN": 1000.0, "STD": 4.0},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 100.0, "STD": 10.0},
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
    assert (tmp_path / 'photometric_precision_vs_dose.npz').is_file()


def test_pixel_precision_analysis_generates_maps(tmp_path):
    bias = tmp_path / 'b.fits'
    dark = tmp_path / 'd.fits'
    _make_fits(bias, 1000)
    _make_fits(dark, 10)

    groups = {
        ('radiating', 'BIAS', 1.0, None): [str(bias)],
        ('radiating', 'DARK', 1.0, None): [str(dark)],
    }

    out_dir = tmp_path / 'out'
    stats = _pixel_precision_analysis(groups, str(out_dir))

    assert (out_dir / 'mag_err_1kR.png').is_file()
    assert (out_dir / 'adu_err16_1kR.png').is_file()
    assert (out_dir / 'adu_err12_1kR.png').is_file()
    assert (out_dir / 'mag_err_vs_dose.png').is_file()
    assert (out_dir / 'adu_err_vs_dose.png').is_file()
    assert (out_dir / 'mag_err_vs_dose.npz').is_file()
    assert (out_dir / 'adu_err_vs_dose.npz').is_file()
    assert set(stats.columns) == {"DOSE", "MAG_MEAN", "MAG_STD", "ADU_MEAN", "ADU_STD"}
    assert len(stats) == 1


def test_fit_dose_response_outputs(tmp_path, monkeypatch):
    summary = pd.DataFrame([
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 2.0, "MEAN": 20.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 2.0, "MEAN": 24.0, "STD": 0.1},
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
    _make_fits(master_dir / "master_bias_radiating_D1kR_E1.0.fits", 2)
    _make_fits(master_dir / "master_bias_radiating_D5kR_E1.0.fits", 5)
    _make_fits(master_dir / "master_bias_post_D5kR_E1.0.fits", 4)

    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.0},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.0},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 5.0, "STD": 0.0},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 4.0, "STD": 0.0},
    ])

    _compare_stage_differences(summary, str(master_dir), str(out_dir))

    assert (out_dir / "stage_differences.csv").is_file()
    assert (out_dir / "bias_first_vs_pre.png").is_file()
    assert (out_dir / "bias_first_vs_pre_log.png").is_file()
    assert (out_dir / "bias_post_vs_last.png").is_file()
    assert (out_dir / "bias_post_vs_last_log.png").is_file()


def test_dynamic_range_analysis_outputs(tmp_path):
    # Use the radiating stage which should be accepted alongside 'during'
    summary = pd.DataFrame([
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 0.0, "MEAN": 10.0, "STD": 1.0},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 5.0, "STD": 2.0},
    ])

    df = _dynamic_range_analysis(summary, str(tmp_path))

    assert (tmp_path / "dynamic_range_vs_dose_16.png").is_file()
    assert (tmp_path / "dynamic_range_vs_dose_12.png").is_file()
    assert (tmp_path / "dynamic_range.npz").is_file()
    assert set(df.columns) == {
        "DOSE",
        "BIAS_MEAN",
        "DARK_MEAN",
        "DR_16",
        "DR_12",
        "NOISE_ADU",
        "NOISE_MAG",
        "RED_16",
        "RED_12",
        "BASE_16",
        "BASE_12",
    }
    row = df.iloc[0]
    expected_noise = np.sqrt(1.0 ** 2 + 2.0 ** 2)
    assert np.isclose(row["BIAS_MEAN"], 10.0)
    assert np.isclose(row["DARK_MEAN"], 5.0)
    assert np.isclose(row["DR_16"], 65536 - 15.0)
    assert np.isclose(row["NOISE_ADU"], expected_noise)
    assert np.isclose(row["RED_16"], 100 * 15.0 / 65536.0)


def test_fit_base_level_trend_outputs(tmp_path, monkeypatch):
    summary = pd.DataFrame([
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 0.1},
    ])

    saved = []

    def fake_savefig(self, path, *a, **k):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_savefig)

    _fit_base_level_trend(summary, tmp_path)

    assert (tmp_path / "base_level_trend.csv").is_file()
    assert sorted(saved) == [
        "base_level_trend_bias.png",
        "base_level_trend_dark.png",
    ]


def test_stage_base_level_diff_outputs(tmp_path):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 14.0, "STD": 0.1},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 13.0, "STD": 0.1},
        {"STAGE": "pre", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 20.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 22.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 25.0, "STD": 0.1},
        {"STAGE": "post", "CALTYPE": "DARK", "DOSE": 5.0, "EXPTIME": 1.0, "MEAN": 24.0, "STD": 0.1},
    ])

    df = _stage_base_level_diff(summary, tmp_path)

    assert (tmp_path / "stage_base_diff.npz").is_file()
    assert (tmp_path / "stage_base_diff_bias.png").is_file()
    assert (tmp_path / "stage_base_diff_dark.png").is_file()

    assert set(df["CALTYPE"]) == {"BIAS", "DARK"}
    assert set(df["CMP"]) == {"first_pre", "last_post"}


def test_relative_precision_analysis_outputs(tmp_path):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 1.0},
        {"STAGE": "pre", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 1.0},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 1.5},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 1.2},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 11.0, "STD": 1.1},
        {"STAGE": "post", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 2.5, "STD": 1.0},
    ])

    df = _relative_precision_analysis(summary, tmp_path)

    assert (tmp_path / "relative_noise_vs_dose_16.png").is_file()
    assert (tmp_path / "relative_mag_err_vs_dose_16.png").is_file()
    assert (tmp_path / "relative_precision.npz").is_file()

    pre_diff = df[df["STAGE"] == "pre"]["NOISE16_DIFF"].iloc[0]
    assert abs(pre_diff) < 1e-6



def test_relative_precision_analysis_npz_and_plots(tmp_path, monkeypatch):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 0.5},
        {"STAGE": "pre", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 11.0, "STD": 0.6},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.2},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 10.5, "STD": 0.55},
        {"STAGE": "post", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.5, "STD": 0.15},
    ])

    saved = []

    def fake_savefig(self, path, *a, **k):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_savefig)

    df = _relative_precision_analysis(summary, tmp_path)

    expected = {
        "relative_noise_vs_dose_16.png",
        "relative_noise_vs_dose_12.png",
        "relative_mag_err_vs_dose_16.png",
        "relative_mag_err_vs_dose_12.png",
    }
    assert expected.issubset(set(saved))
    for name in expected:
        assert (tmp_path / name.replace(".png", ".npz")).is_file()
    assert (tmp_path / "pre_vs_post_relative_precision.npz").is_file()
    assert (tmp_path / "relative_precision.npz").is_file()
    assert len(df) == 3
    assert df[df["STAGE"] == "pre"]["NOISE16_DIFF"].iloc[0] == 0.0


def test_relative_precision_analysis_minimal(tmp_path, monkeypatch):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 0.5},
        {"STAGE": "pre", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 11.0, "STD": 0.6},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.2},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.5, "STD": 0.55},
        {"STAGE": "post", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 2.5, "STD": 0.15},
    ])

    saved = []

    def fake_savefig(self, path, *a, **k):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_savefig)

    df = _relative_precision_analysis(summary, tmp_path)

    expected_pngs = {
        "relative_noise_vs_dose_16.png",
        "relative_noise_vs_dose_12.png",
        "relative_mag_err_vs_dose_16.png",
        "relative_mag_err_vs_dose_12.png",
    }
    assert expected_pngs.issubset(set(saved))
    for name in expected_pngs:
        assert (tmp_path / name.replace(".png", ".npz")).is_file()
    assert (tmp_path / "relative_precision.npz").is_file()
    assert (tmp_path / "pre_vs_post_relative_precision.npz").is_file()
    assert len(df) == 3


def test_plot_bias_dark_error_outputs(tmp_path):
    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.3},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.2},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 2.5, "STD": 0.25},
        {"STAGE": "pre", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 1.0},
        {"STAGE": "post", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 15.0, "STD": 1.5},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 1.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 14.0, "STD": 1.4},
    ])

    _plot_bias_dark_error(summary, tmp_path)

    # Bias files
    b_png = tmp_path / "bias_mean_std_vs_dose.png"
    b_npz = tmp_path / "bias_mean_std_vs_dose.npz"
    assert b_png.is_file() and b_npz.is_file()
    b_data = np.load(b_npz)
    assert "slope_ir" in b_data and "slope_no" in b_data
    assert (tmp_path / "std_model_bias.png").is_file()

    # Dark files include exposure time
    d_png = tmp_path / "dark_mean_std_vs_dose_E1p0s.png"
    d_npz = tmp_path / "dark_mean_std_vs_dose_E1p0s.npz"
    assert d_png.is_file() and d_npz.is_file()
    d_data = np.load(d_npz)
    assert "slope_ir" in d_data and "slope_no" in d_data
    assert (tmp_path / "std_model_dark_E1p0s.png").is_file()


def test_estimate_dose_rate_and_plot(tmp_path):
    d1 = tmp_path / "1kRads"
    d2 = tmp_path / "2kRads"
    d1.mkdir()
    d2.mkdir()

    f1 = d1 / "f1.fits"
    f2 = d1 / "f2.fits"
    f3 = d2 / "f1.fits"
    f4 = d2 / "f2.fits"
    _make_fits(f1, 1, ts=0)
    _make_fits(f2, 1, ts=10)
    _make_fits(f3, 2, ts=20)
    _make_fits(f4, 2, ts=40)

    df = pd.DataFrame({
        "PATH": [str(f1), str(f2), str(f3), str(f4)],
        "CALTYPE": ["BIAS"] * 4,
        "STAGE": ["radiating"] * 4,
        "VACUUM": ["air"] * 4,
        "TEMP": [10.0] * 4,
        "ZEROFRACTION": [0.0] * 4,
        "BADFITS": [False] * 4,
    })

    rate_df = _estimate_dose_rate(df)
    assert len(rate_df) == 2
    r1 = rate_df.sort_values("DOSE")["DOSE_RATE"].iloc[0]
    r2 = rate_df.sort_values("DOSE")["DOSE_RATE"].iloc[1]
    assert np.isclose(r1, 0.0)
    assert np.isclose(r2, 0.05)
    assert "DOSE_RATE_STD" in rate_df.columns

    summary = pd.DataFrame([
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.2, "DOSE_RATE": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.3, "DOSE_RATE": 0.05},
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1, "DOSE_RATE": np.nan},
    ])

    _plot_dose_rate_effect(summary, tmp_path)

    png = tmp_path / "dose_rate_effect_bias.png"
    npz = tmp_path / "dose_rate_effect_bias.npz"
    assert png.is_file()
    assert npz.is_file()
    data = np.load(npz)
    assert "slope_mean_ir" in data
    assert np.isclose(data["slope_mean_ir"], -20)
    assert np.isclose(data["intercept_mean_ir"], 4)
    assert np.isclose(data["slope_std_ir"], -2)
    assert np.isclose(data["intercept_std_ir"], 0.4)


def test_estimate_dose_rate_hierarch_timestamp(tmp_path):
    ddir = tmp_path / "1kR"
    ddir.mkdir()
    f1 = ddir / "f1.fits"
    f2 = ddir / "f2.fits"
    _make_fits(f1, 1)
    _make_fits(f2, 1)
    with fits.open(f1, mode="update") as h:
        h[0].header.pop("TIMESTAMP", None)
        h[0].header["HIERARCH TIMESTAMP"] = 0
    with fits.open(f2, mode="update") as h:
        h[0].header.pop("TIMESTAMP", None)
        h[0].header["HIERARCH TIMESTAMP"] = 10

    df = pd.DataFrame({
        "PATH": [str(f1), str(f2)],
        "CALTYPE": ["BIAS", "BIAS"],
        "STAGE": ["radiating", "radiating"],
        "VACUUM": ["air", "air"],
        "TEMP": [10.0, 10.0],
        "ZEROFRACTION": [0.0, 0.0],
        "BADFITS": [False, False],
    })

    rate_df = _estimate_dose_rate(df)
    assert np.isclose(rate_df["DOSE_RATE"].iloc[0], 0.0)
    assert "DOSE_RATE_STD" in rate_df.columns


def test_dose_rate_full_pipeline(tmp_path):
    """Estimate dose rate and generate plots for bias and dark."""
    d1 = tmp_path / "1kRads"
    d2 = tmp_path / "2kRads"
    d1.mkdir()
    d2.mkdir()

    paths = []
    caltypes = []
    stages = []
    for dose, ddir, start in [(1.0, d1, 0), (2.0, d2, 20)]:
        for cal in ["BIAS", "DARK"]:
            f1 = ddir / f"{cal.lower()}_1.fits"
            f2 = ddir / f"{cal.lower()}_2.fits"
            _make_fits(f1, 1, ts=start)
            _make_fits(f2, 1, ts=start + (10 if dose == 1.0 else 20))
            paths.extend([f1, f2])
            caltypes.extend([cal, cal])
            stages.extend(["radiating", "radiating"])

    df = pd.DataFrame({
        "PATH": [str(p) for p in paths],
        "CALTYPE": caltypes,
        "STAGE": stages,
        "VACUUM": ["air"] * len(paths),
        "TEMP": [10.0] * len(paths),
        "ZEROFRACTION": [0.0] * len(paths),
        "BADFITS": [False] * len(paths),
    })

    rate_df = _estimate_dose_rate(df)
    for dose, expected in [(1.0, 0.0), (2.0, 0.05)]:
        vals = rate_df[rate_df["DOSE"] == dose]["DOSE_RATE"].dropna().unique()
        assert len(vals) == 1
        assert np.isclose(vals[0], expected)
    assert "DOSE_RATE_STD" in rate_df.columns

    summary = pd.DataFrame([
        {"STAGE": "pre", "CALTYPE": "BIAS", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 1.0, "STD": 0.1},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 2.0, "STD": 0.2},
        {"STAGE": "radiating", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 3.0, "STD": 0.3},
        {"STAGE": "post", "CALTYPE": "BIAS", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 4.0, "STD": 0.4},
        {"STAGE": "pre", "CALTYPE": "DARK", "DOSE": 0.0, "EXPTIME": 1.0, "MEAN": 10.0, "STD": 1.0},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 1.0, "EXPTIME": 1.0, "MEAN": 11.0, "STD": 1.1},
        {"STAGE": "radiating", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 12.0, "STD": 1.2},
        {"STAGE": "post", "CALTYPE": "DARK", "DOSE": 2.0, "EXPTIME": 1.0, "MEAN": 13.0, "STD": 1.3},
    ])
    summary = summary.merge(rate_df, on=["STAGE", "CALTYPE", "DOSE", "EXPTIME"], how="left")

    _plot_bias_dark_error(summary, tmp_path)
    _plot_dose_rate_effect(summary, tmp_path)

    assert (tmp_path / "bias_mean_std_vs_dose.png").is_file()
    assert (tmp_path / "dark_mean_std_vs_dose_E1p0s.png").is_file()
    assert (tmp_path / "std_model_bias.png").is_file()
    assert (tmp_path / "std_model_dark_E1p0s.png").is_file()
    for cal in ("bias",):
        assert (tmp_path / f"dose_rate_effect_{cal}.png").is_file()
        npz = tmp_path / f"dose_rate_effect_{cal}.npz"
        assert npz.is_file()
        data = np.load(npz)
        for key in [
            "slope_mean_ir",
            "intercept_mean_ir",
            "slope_std_ir",
            "intercept_std_ir",
        ]:
            assert key in data

