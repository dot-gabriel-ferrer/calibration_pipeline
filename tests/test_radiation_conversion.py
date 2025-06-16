import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits

from run_full_radiation_pipeline import _ensure_conversion


def _make_fits(path):
    fits.writeto(path, np.zeros((2, 2), dtype=np.float32), overwrite=True)


def test_missing_radiation_logs(tmp_path):
    root = tmp_path
    for dose in ("0kRads", "10kRads"):
        fdir = root / dose / "fits"
        fdir.mkdir(parents=True)
        for i in range(2):
            _make_fits(fdir / f"f{i}.fits")

    _ensure_conversion(str(root))

    df = pd.read_csv(root / "radiationLogCompleto.csv")
    assert list(df.columns) == ["FrameNum", "Dose"]
    assert df["Dose"].tolist() == [0.0, 0.0, 5.0, 10.0]


def test_constant_radiation_log(tmp_path):
    root = tmp_path
    for dose in ("10kRads", "20kRads"):
        dpath = root / dose
        fdir = dpath / "fits"
        fdir.mkdir(parents=True)
        for i in range(2):
            _make_fits(fdir / f"f{i}.fits")
        pd.DataFrame({"FrameNum": [0, 1], "RadiationLevel": [1.0, 1.0]}).to_csv(
            dpath / "radiationLogDef.csv", index=False
        )

    _ensure_conversion(str(root))

    df = pd.read_csv(root / "radiationLogCompleto.csv")
    assert "RadiationLevel" in df.columns and "Dose" not in df.columns
    assert df["RadiationLevel"].tolist() == [5.0, 10.0, 15.0, 20.0]
