import os
import numpy as np
from astropy.io import fits
import pytest

from dark_pipeline.dark_pipeline.steps.outgasing_destruction_analysis import (
    generate_bias_maps,
    generate_dark_maps,
    find_closest_dark,
    classify_outliers,
)


def _make_fits(path, data, temp=None, exp=None):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    if temp is not None:
        hdu.header['TEMPERATURE'] = temp
        hdu.header['temperature'] = temp
    if exp is not None:
        hdu.header['EXPOSURE'] = exp
        hdu.header['exposure'] = exp
    hdu.writeto(path, overwrite=True)


@pytest.fixture
def sample_dataset(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def write(name, arr, temp=None, exp=None):
        path = data_dir / name
        _make_fits(path, arr, temp=temp, exp=exp)
        return str(path)

    bias_files = [
        {"original_path": write("b_t10_1.fits", np.full((2, 2), 10), temp=10.0), "temperature": 10.0},
        {"original_path": write("b_t10_2.fits", np.full((2, 2), 12), temp=10.0), "temperature": 10.0},
        {"original_path": write("b_t20_1.fits", np.full((2, 2), 20), temp=20.0), "temperature": 20.0},
        {"original_path": write("b_t20_2.fits", np.full((2, 2), 22), temp=20.0), "temperature": 20.0},
    ]

    dark_files = [
        {"original_path": write("d_t10_e1_1.fits", np.full((2, 2), 100), temp=10.0, exp=1.0),
         "temperature": 10.0, "exposure_time": 1.0},
        {"original_path": write("d_t10_e1_2.fits", np.full((2, 2), 102), temp=10.0, exp=1.0),
         "temperature": 10.0, "exposure_time": 1.0},
        {"original_path": write("d_t20_e1_1.fits", np.full((2, 2), 200), temp=20.0, exp=1.0),
         "temperature": 20.0, "exposure_time": 1.0},
        {"original_path": write("d_t20_e2_1.fits", np.full((2, 2), 400), temp=20.0, exp=2.0),
         "temperature": 20.0, "exposure_time": 2.0},
    ]

    return bias_files, dark_files


def test_generate_bias_maps(sample_dataset):
    bias_files, _ = sample_dataset
    maps = generate_bias_maps(bias_files)
    assert set(maps.keys()) == {10.0, 20.0}
    assert np.allclose(maps[10.0], np.full((2, 2), 11))
    assert np.allclose(maps[20.0], np.full((2, 2), 21))


def test_generate_dark_maps_and_find_closest(sample_dataset):
    _, dark_files = sample_dataset
    maps = generate_dark_maps(dark_files)
    assert np.allclose(maps[(10.0, 1.0)], np.full((2, 2), 101))
    assert np.allclose(maps[(20.0, 1.0)], np.full((2, 2), 200))
    assert np.allclose(maps[(20.0, 2.0)], np.full((2, 2), 400))

    closest = find_closest_dark(20.1, 2.05, maps, temp_tolerance=0.2, exp_tolerance=0.1)
    assert np.allclose(closest, np.full((2, 2), 400))


def test_classify_outliers_pre_post():
    pre = {
        (0, 0): {('Pre', 't1')},
        (1, 1): {('Pre', 't1'), ('Pre', 't2')},
        (2, 2): {('Pre', 't1'), ('Pre', 't2')},
        (3, 3): {('Pre', 't1')},
    }
    during = {
        (0, 0): {('During', 't3')},
        (1, 1): {('During', 't3')},
    }
    post = {
        (0, 0): {('Post', 't4')},
    }

    results = classify_outliers([pre, during, post])
    class_dict = {tuple(r['pixel']): r['classification'] for r in results}

    assert class_dict[(0, 0)] == 'Sensor Damage'
    assert class_dict[(1, 1)] == 'Sensor Damage'
    assert class_dict[(2, 2)] == 'Likely Sensor'
    assert class_dict[(3, 3)] == 'Temperature'
