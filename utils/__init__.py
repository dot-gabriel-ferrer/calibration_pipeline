import os
import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


def read_radiation_log(path: str) -> Dict[int, float]:
    """Return mapping from ``FrameNum`` to radiation dose.

    Parameters
    ----------
    path : str
        CSV file produced by the irradiation setup.

    Returns
    -------
    dict[int, float]
        Mapping of frame number to dose. Empty if the file cannot be read.
    """
    if not os.path.isfile(path):
        logger.warning("Radiation log %s not found", path)
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read radiation log %s: %s", path, exc)
        return {}

    frame = pd.to_numeric(df.get("FrameNum"), errors="coerce")
    if frame.isna().all():
        frame = pd.Series(range(len(df)))

    dose_col = "Dose" if "Dose" in df.columns else "RadiationLevel"
    dose = pd.to_numeric(df.get(dose_col), errors="coerce")

    mapping: Dict[int, float] = {}
    for fr, d in zip(frame, dose):
        if pd.notna(fr) and pd.notna(d):
            mapping[int(fr)] = float(d)
    return mapping
