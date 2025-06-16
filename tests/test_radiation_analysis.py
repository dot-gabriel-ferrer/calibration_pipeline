import numpy as np
from radiation_analysis import diff_heatmap, analyse_stage
from astropy.io import fits
import pandas as pd


def _make_fits(path, value, temp=10.0, frame=None):
    h = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    h.header["TEMP"] = temp
    if frame is not None:
        h.header["FRAMENUM"] = frame
    h.writeto(path, overwrite=True)


def test_diff_heatmap_saves_npz(tmp_path):
    ref = np.zeros((2, 2), dtype=float)
    targ = np.ones((2, 2), dtype=float)
    out_png = tmp_path / "diff.png"
    diff_heatmap(ref, targ, str(out_png), "title")
    assert (tmp_path / "diff.npz").is_file()


def test_analyse_stage_bias_subtraction(tmp_path):
    b1 = tmp_path / "b1.fits"
    b2 = tmp_path / "b2.fits"
    d1 = tmp_path / "d1.fits"
    d2 = tmp_path / "d2.fits"

    _make_fits(b1, 2, temp=10.0)
    _make_fits(b2, 2, temp=10.0)
    _make_fits(d1, 102, temp=10.0, frame=1)
    _make_fits(d2, 104, temp=10.0, frame=2)

    df = pd.DataFrame(
        {
            "PATH": [str(b1), str(b2), str(d1), str(d2)],
            "CALTYPE": ["BIAS", "BIAS", "DARK", "DARK"],
            "STAGE": ["pre"] * 4,
        }
    )

    outdir = tmp_path / "out"
    analyse_stage(df, "missing.csv", str(outdir), "pre")

    stats = pd.read_csv(outdir / "stats_dark.csv")
    assert list(stats["MEAN"]) == [100.0, 102.0]

    hdr = fits.getheader(outdir / "master_dark_T10.0.fits")
    assert "HISTORY" in hdr and "Bias-corrected" in hdr["HISTORY"]
