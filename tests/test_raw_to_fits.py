import numpy as np
from astropy.io import fits
from utils.raw_to_fits import convert_attempt, parse_frame_number


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
