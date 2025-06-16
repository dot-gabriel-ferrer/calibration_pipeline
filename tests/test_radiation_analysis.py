import numpy as np
import pandas as pd
from astropy.io import fits
from radiation_analysis import diff_heatmap, plot_mean_std_vs_time, main as rad_main


def test_diff_heatmap_saves_npz(tmp_path):
    ref = np.zeros((2, 2), dtype=float)
    targ = np.ones((2, 2), dtype=float)
    out_png = tmp_path / "diff.png"
    diff_heatmap(ref, targ, str(out_png), "title")
    assert (tmp_path / "diff.npz").is_file()


def test_plot_mean_std_vs_time_creates_png(tmp_path):
    df = pd.DataFrame(
        {
            "TEMP": [10.0, 10.0, 20.0, 20.0],
            "FRAME": [0, 1, 0, 1],
            "MEAN": [1.0, 1.1, 2.0, 2.2],
            "STD": [0.1, 0.1, 0.2, 0.2],
        }
    )
    csv = tmp_path / "stats.csv"
    df.to_csv(csv, index=False)
    out_png = tmp_path / "plot.png"
    plot_mean_std_vs_time(str(csv), str(out_png))
    assert out_png.is_file()


def _make_fits(path, value, temp=10.0, exp=1.0, frame=0):
    hdu = fits.PrimaryHDU(np.full((2, 2), value, dtype=np.float32))
    hdu.header["TEMP"] = temp
    hdu.header["EXPTIME"] = exp
    hdu.header["FRAMENUM"] = frame
    hdu.writeto(path, overwrite=True)


def test_group_by_dose_constant_dataset(tmp_path):
    d1 = tmp_path / "d1.fits"
    d2 = tmp_path / "d2.fits"
    _make_fits(d1, 1, temp=10.0, exp=1.0, frame=0)
    _make_fits(d2, 2, temp=20.0, exp=1.0, frame=1)

    index_csv = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "PATH": [str(d1), str(d2)],
            "CALTYPE": ["DARK", "DARK"],
            "STAGE": ["pre", "pre"],
            "VACUUM": ["air", "air"],
            "TEMP": [10.0, 20.0],
            "ZEROFRACTION": [0.0, 0.0],
            "BADFITS": [False, False],
        }
    ).to_csv(index_csv, index=False)

    rad_csv = tmp_path / "rad.csv"
    pd.DataFrame({"FrameNum": [0, 1], "RadiationLevel": [0, 0]}).to_csv(rad_csv, index=False)

    out_dir = tmp_path / "out"
    rad_main(str(index_csv), str(rad_csv), str(out_dir), ["pre"], group_by_dose=True)

    master = out_dir / "pre" / "master_dark_T0.0.fits"
    assert master.is_file()
    # no temperature-separated masters should exist
    assert not (out_dir / "pre" / "master_dark_T10.0.fits").exists()
    assert not (out_dir / "pre" / "master_dark_T20.0.fits").exists()
