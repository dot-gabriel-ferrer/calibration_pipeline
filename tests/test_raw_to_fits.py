import csv
import numpy as np
import pytest
from astropy.io import fits
from utils.raw_to_fits import convert_attempt, convert_many, parse_frame_number


def test_convert_attempt_parses_exptime(tmp_path):
    attempt = tmp_path / "attempt0"
    frames = attempt / "frames"
    frames.mkdir(parents=True)

    with open(attempt / "configFile.txt", "w") as f:
        f.write("WIDTH: 2\nHEIGHT: 2\nBIT_DEPTH: 16\n")

    with open(attempt / "temperatureLog.csv", "w") as f:
        f.write(
            "FrameNum,TimeStamp,ExtTemperature,ExpTime,RealExpTime,ExpGain,Temperature,InitialTemp,DeltaTemperature,PowerCons\n"
        )
        f.write("0,100,0,12,12,1,-0.5,-0.5,0,50\n")
        f.write("1,101,0,12,12,1,-0.4,-0.5,0,50\n")

    arr0 = np.arange(4, dtype=np.uint16).reshape(2, 2)
    arr0.tofile(frames / "BiasT0_exp0.1sAt0f0.raw")

    arr1 = np.arange(4, 8, dtype=np.uint16).reshape(2, 2)
    arr1.tofile(frames / "BiasT0_exp0.1sAt0f1.raw")

    fits_files = convert_attempt(str(attempt), "BIAS")
    assert len(fits_files) == 2

    hdr = fits.getheader(fits_files[0])
    assert hdr["FRAMENUM"] == 0
    assert hdr["TIMESTAMP"] == 100.0
    assert hdr["EXPTIME"] == 12 / 1e6
    assert hdr["TEMP"] == -0.5
    assert hdr["FILETEMP"] == 0


def test_convert_attempt_custom_headers(tmp_path):
    attempt = tmp_path / "attempt0"
    frames = attempt / "frames"
    frames.mkdir(parents=True)

    with open(attempt / "configFile.txt", "w") as f:
        f.write("WIDTH: 2\nHEIGHT: 2\nBIT_DEPTH: 16\n")

    with open(attempt / "temperatureLog.csv", "w") as f:
        f.write(
            "FrameNum,TimeStamp,ExtTemperature,ExpTime,RealExpTime,ExpGain,Temperature,InitialTemp\n"
        )
        f.write("0,100,5,12,12,2,-10,-11\n")

    arr0 = np.arange(4, dtype=np.uint16).reshape(2, 2)
    arr0.tofile(frames / "f0.raw")

    fits_file = convert_attempt(str(attempt), "BIAS")[0]
    hdr = fits.getheader(fits_file)
    assert hdr["GAIN"] == 2
    assert hdr["TEMP"] == -10
    assert hdr["TEMP_0"] == -11
    assert hdr["EQTEMP"] == 5


def test_parse_frame_number_frame_prefix():
    assert parse_frame_number("exp_1.2e-05s_frame0.raw") == 0


@pytest.mark.parametrize(
    "skip_flag,caltype",
    [
        ("skip_bias", "BIAS"),
        ("skip_dark", "DARK"),
        ("skip_flat", "FLAT"),
    ],
)
def test_convert_many_skip_flags(tmp_path, skip_flag, caltype):
    bias_root = tmp_path / "bias"
    dark_root = tmp_path / "dark"
    flat_root = tmp_path / "flat"

    for root in (bias_root, dark_root, flat_root):
        attempt = root / "T0" / "attempt0" / "frames"
        attempt.mkdir(parents=True)
        with open(attempt.parent / "configFile.txt", "w") as f:
            f.write("WIDTH: 1\nHEIGHT: 1\nBIT_DEPTH: 16\n")
        with open(attempt.parent / "temperatureLog.csv", "w") as f:
            f.write("FrameNum\n0\n")
        np.array([1], dtype=np.uint16).tofile(attempt / "f0.raw")

    kwargs = {"skip_bias": False, "skip_dark": False, "skip_flat": False}
    kwargs[skip_flag] = True

    convert_many(
        str(bias_root),
        str(dark_root),
        str(flat_root),
        search_depth=2,
        **kwargs,
    )

    paths = {
        "BIAS": bias_root / "T0" / "attempt0" / "fits",
        "DARK": dark_root / "T0" / "attempt0" / "fits",
        "FLAT": flat_root / "T0" / "attempt0" / "fits",
    }

    # skipped dataset should not have a fits directory
    assert not paths[caltype].exists()
    # other datasets should have a fits directory
    for ct, p in paths.items():
        if ct != caltype:
            assert p.is_dir()

    # verify index CSV does not contain the skipped calibration type
    csv_path = tmp_path / "fits_index.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert all(row["CALTYPE"] != caltype for row in rows)


def test_convert_many_deep_dark(tmp_path):
    """convert_many handles deep dark directory structures"""
    bias_root = tmp_path / "TestSection1"
    dark_root = tmp_path / "TestSection2"
    flat_root = tmp_path / "TestSection3"

    attempt = dark_root / "20Frames" / "T20" / "T0" / "0.4s" / "attempt0"
    frames = attempt / "frames"
    frames.mkdir(parents=True)

    with open(attempt / "configFile.txt", "w") as f:
        f.write("WIDTH: 1\nHEIGHT: 1\nBIT_DEPTH: 16\n")
    with open(attempt / "temperatureLog.csv", "w") as f:
        f.write("FrameNum\n0\n")
    np.array([1], dtype=np.uint16).tofile(frames / "f0.raw")

    convert_many(
        str(bias_root),
        str(dark_root),
        str(flat_root),
        search_depth=6,
        skip_bias=True,
    )

    fits_file = attempt / "fits" / "f0.fits"
    assert fits_file.is_file()
