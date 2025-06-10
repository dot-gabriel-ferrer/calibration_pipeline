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


def test_convert_many_frames_without_attempt_dirs(tmp_path):
    bias_root = tmp_path / "bias"
    dark_root = tmp_path / "dark"
    flat_root = tmp_path / "flat"

    for root in (bias_root, dark_root, flat_root):
        frames = root / "20Frames" / "T20" / "T0" / "0.4s" / "frames"
        frames.mkdir(parents=True)
        with open(frames.parent / "configFile.txt", "w") as f:
            f.write("WIDTH: 1\nHEIGHT: 1\nBIT_DEPTH: 16\n")
        with open(frames.parent / "temperatureLog.csv", "w") as f:
            f.write("FrameNum\n0\n")
        np.array([1], dtype=np.uint16).tofile(frames / "f0.raw")

    convert_many(
        str(bias_root),
        str(dark_root),
        str(flat_root),
        search_depth=6,
        skip_bias=True,
    )

    for root in (dark_root, flat_root):
        assert (root / "20Frames" / "T20" / "T0" / "0.4s" / "fits").is_dir()
    # bias section was skipped
    assert not (bias_root / "20Frames" / "T20" / "T0" / "0.4s" / "fits").exists()

    csv_path = tmp_path / "fits_index.csv"
    assert csv_path.is_file()


def test_incomplete_raw_is_padded_and_flagged(tmp_path):
    bias_root = tmp_path / "bias"
    frames = bias_root / "T0" / "attempt0" / "frames"
    frames.mkdir(parents=True)

    with open(frames.parent / "configFile.txt", "w") as f:
        f.write("WIDTH: 2\nHEIGHT: 2\nBIT_DEPTH: 16\n")

    with open(frames.parent / "temperatureLog.csv", "w") as f:
        f.write("FrameNum\n0\n")

    # write a raw file with fewer pixels than expected
    np.array([1, 2, 3], dtype=np.uint16).tofile(frames / "f0.raw")

    convert_many(
        str(bias_root),
        str(bias_root),
        str(bias_root),
        search_depth=2,
        skip_dark=True,
        skip_flat=True,
    )

    fits_path = frames.parent / "fits" / "f0.fits"
    data = fits.getdata(fits_path)
    assert data.shape == (2, 2)
    # last pixel should be padded with zero
    assert data[1, 1] == 0
    hdr = fits.getheader(fits_path)
    assert hdr["BADSIZE"] is True

    csv_path = bias_root / "fits_index.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["BADSIZE"] == "True"
