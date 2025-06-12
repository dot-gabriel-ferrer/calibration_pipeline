#!/usr/bin/env python
"""Run bias/dark variation analysis across irradiation stages.

The script ensures that each calibration attempt has converted FITS files and
then uses ``utils.index_dataset`` and ``radiation_analysis`` to analyse
pre-irradiation, irradiation and post-irradiation datasets.

Usage
-----
    python radiation_variation_analysis.py <dataset_root> <output_dir>

The dataset is expected to contain ``Preirradiation``, ``Irradiation`` and
``Postirradiation`` folders following this structure::

    <dataset_root>/Preirradiation/
        Bias/
        Darks/
        Flats/
    <dataset_root>/Irradiation/<dose>kRads/
        Bias/
        Darks/
        Flats/
    <dataset_root>/Postirradiation/
        Bias/
        Darks/
        Flats/

Each attempt directory must provide ``frames/`` with ``.raw`` files as well as a
``configFile.txt`` and ``radiationLog.csv`` (or ``radiationLogCompleto.csv``).
If the ``fits/`` folder is missing it will be generated automatically.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import pandas as pd

from utils import raw_to_fits, index_dataset
import radiation_analysis


def _ensure_fits(section: str, caltype: str) -> None:
    """Convert raw frames within *section* if no FITS files are present."""
    attempts = raw_to_fits.gather_attempts(section, max_depth=6)
    for attempt in attempts:
        fits_dir = os.path.join(attempt, "fits")
        has_fits = os.path.isdir(fits_dir) and any(
            name.endswith(".fits") for name in os.listdir(fits_dir)
        )
        if not has_fits:
            raw_to_fits.convert_attempt(attempt, caltype)


def _process_stage(root: str, stage: str) -> pd.DataFrame:
    bias = os.path.join(root, "Bias")
    dark = os.path.join(root, "Darks")
    flat = os.path.join(root, "Flats")

    for path, ct in ((bias, "BIAS"), (dark, "DARK"), (flat, "FLAT")):
        if os.path.isdir(path):
            _ensure_fits(path, ct)

    tmp_csv = os.path.join(root, "index.csv")
    index_dataset.index_sections(bias, dark, flat, tmp_csv, stage=stage, search_depth=6)
    if os.path.isfile(tmp_csv):
        return pd.read_csv(tmp_csv)
    return pd.DataFrame()


def main(dataset_root: str, output_dir: str) -> None:
    stages: List[Tuple[str, str]] = []
    pre = os.path.join(dataset_root, "Preirradiation")
    if os.path.isdir(pre):
        stages.append((pre, "pre"))

    irrad_root = os.path.join(dataset_root, "Irradiation")
    if os.path.isdir(irrad_root):
        for name in sorted(os.listdir(irrad_root)):
            path = os.path.join(irrad_root, name)
            if os.path.isdir(path):
                stages.append((path, "during"))

    post = os.path.join(dataset_root, "Postirradiation")
    if os.path.isdir(post):
        stages.append((post, "post"))

    frames = []
    for root, stage in stages:
        frames.append(_process_stage(root, stage))

    if not frames:
        print("No calibration data found")
        return

    index_df = pd.concat(frames, ignore_index=True)
    index_csv = os.path.join(dataset_root, "index.csv")
    index_df.to_csv(index_csv, index=False)

    rad_log = os.path.join(dataset_root, "radiationLog.csv")
    if not os.path.isfile(rad_log):
        rad_log = ""

    radiation_analysis.main(index_csv, rad_log, output_dir, stages=["pre", "during", "post"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse bias/dark variation across irradiation stages")
    parser.add_argument("dataset_root", help="Path to dataset root")
    parser.add_argument("output_dir", help="Directory for analysis results")
    args = parser.parse_args()
    main(args.dataset_root, args.output_dir)
