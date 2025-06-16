import csv
import numpy as np
from pathlib import Path
from astropy.io import fits

from process_index import main as process_main
from dose_analysis import main as dose_main
from radiation_analysis import main as rad_main


def _make_fits(path, value, temp=10.0, exp=1.0):
    hdu = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    hdu.header['TEMP'] = temp
    hdu.header['EXPTIME'] = exp
    hdu.writeto(path, overwrite=True)


def _write_index(bias_paths, dark_paths, csv_path):
    fieldnames = [
        "PATH",
        "CALTYPE",
        "STAGE",
        "VACUUM",
        "TEMP",
        "ZEROFRACTION",
        "BADFITS",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in bias_paths:
            w.writerow({
                "PATH": str(p),
                "CALTYPE": "BIAS",
                "STAGE": "pre",
                "VACUUM": "air",
                "TEMP": 10.0,
                "ZEROFRACTION": 0.0,
                "BADFITS": False,
            })
        for p in dark_paths:
            w.writerow({
                "PATH": str(p),
                "CALTYPE": "DARK",
                "STAGE": "pre",
                "VACUUM": "air",
                "TEMP": 10.0,
                "ZEROFRACTION": 0.0,
                "BADFITS": False,
            })


def test_entrypoints_remove_bias(tmp_path):
    import tempfile

    data_root = Path(tempfile.mkdtemp(prefix="ds_")) / "run1"
    bdir = data_root / "bias" / "fits"
    ddir = data_root / "dark" / "fits"
    bdir.mkdir(parents=True)
    ddir.mkdir(parents=True)

    b1 = bdir / "b1.fits"
    b2 = bdir / "b2.fits"
    d1 = ddir / "d1.fits"
    d2 = ddir / "d2.fits"

    _make_fits(b1, 2, temp=10.0, exp=0.0)
    _make_fits(b2, 2, temp=10.0, exp=0.0)
    _make_fits(d1, 7, temp=10.0, exp=1.0)
    _make_fits(d2, 7, temp=10.0, exp=1.0)

    index_csv = tmp_path / "index.csv"
    _write_index([b1, b2], [d1, d2], index_csv)

    out_proc = tmp_path / "out_proc"
    process_main(str(index_csv), str(out_proc))
    mdark = fits.getdata(out_proc / "masters" / "darks" / "master_dark_T10.0_E1.0.fits")
    assert np.allclose(mdark, np.full((2, 2), 5.0))

    out_dose = tmp_path / "out_dose"
    dose_main(str(index_csv), str(out_dose), verbose=False)
    mdark = fits.getdata(out_dose / "masters" / "master_dark_pre_D0kR_E1.0.fits")
    assert np.allclose(mdark, np.full((2, 2), 5.0))

    rad_log = tmp_path / "rad.csv"
    rad_log.write_text("FrameNum,RadiationLevel\n")
    out_rad = tmp_path / "out_rad"
    rad_main(str(index_csv), str(rad_log), str(out_rad), ["pre"])
    mdark = fits.getdata(out_rad / "pre" / "master_dark_T10.0.fits")
    assert np.allclose(mdark, np.full((2, 2), 5.0))
