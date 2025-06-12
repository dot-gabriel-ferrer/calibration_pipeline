import numpy as np
from astropy.io import fits


import pandas as pd
from operation_analysis import _load_frames, main, _plot_rad_vs_outliers

import os


def test_load_frames_numeric_order(tmp_path):
    attempt = tmp_path / "attempt"
    fits_dir = attempt / "fits"
    fits_dir.mkdir(parents=True)

    # create files in non-numeric order
    names = ["f10.fits", "f2.fits", "f1.fits"]
    for name in names:
        fits.writeto(fits_dir / name, np.zeros((1, 1), dtype=np.float32), overwrite=True)

    expected = [
        str(fits_dir / "f1.fits"),
        str(fits_dir / "f2.fits"),
        str(fits_dir / "f10.fits"),
    ]
    assert _load_frames(str(attempt)) == expected


def test_main_dataset_filter(monkeypatch, tmp_path):
    in_dir = tmp_path / "Operation"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "10kRads").mkdir()
    (in_dir / "20kRads").mkdir()

    called = []

    def fake_analyze_directory(path, out):
        called.append(os.path.basename(path))

    monkeypatch.setattr("operation_analysis.analyze_directory", fake_analyze_directory)

    main(str(in_dir), str(out_dir), datasets=["20kRads"])

    assert called == ["20kRads"]


def test_plot_rad_vs_outliers_handles_failed_polyfit(monkeypatch, tmp_path):
    rad_df = pd.DataFrame({"TimeStamp": [0, 1], "Dose": [0, 1]})
    frame_times = [0, 1]
    outlier_counts = [1, 2]
    outpath = tmp_path / "plot.png"

    def raise_linalg_err(*args, **kwargs):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np, "polyfit", raise_linalg_err)

    _plot_rad_vs_outliers(rad_df, frame_times, outlier_counts, str(outpath))

    assert outpath.is_file()

