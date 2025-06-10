import numpy as np
from astropy.io import fits
from utils.raw_to_fits import convert_attempt


def test_convert_attempt_parses_exptime(tmp_path):
    attempt = tmp_path / "attempt0"
    frames = attempt / "frames"
    frames.mkdir(parents=True)

    with open(attempt / "configFile.txt", "w") as f:
        f.write("WIDTH: 2\nHEIGHT: 2\nBIT_DEPTH: 16\n")

    with open(attempt / "temperatureLog.csv", "w") as f:
        f.write("FrameNum,Temperature\n0,10\n1,11\n")

    arr0 = np.arange(4, dtype=np.uint16).reshape(2, 2)
    arr0.tofile(frames / "BiasT0_exp0.1sAt0f0.raw")

    arr1 = np.arange(4, 8, dtype=np.uint16).reshape(2, 2)
    arr1.tofile(frames / "BiasT0_exp0.1sAt0f1.raw")

    fits_files = convert_attempt(str(attempt), "BIAS")
    assert len(fits_files) == 2

    hdr = fits.getheader(fits_files[0])
    assert hdr["EXPTIME"] == 0.1
    assert hdr["CCD_TEMP"] == 10
    assert hdr["TEMP"] == 10
    assert hdr["FILETEMP"] == 0
