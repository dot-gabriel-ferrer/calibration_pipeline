#!/usr/bin/env python
"""Convert, index and analyse irradiation datasets."""

from __future__ import annotations

import argparse
import os
import pandas as pd

from utils import raw_to_fits, index_dataset
import radiation_analysis


def _gather_stages(base: str) -> list[tuple[str, str]]:
    stages: list[tuple[str, str]] = []
    pre = os.path.join(base, "Preirradiation")
    if os.path.isdir(pre):
        stages.append((pre, "pre"))

    irrad_root = os.path.join(base, "Irradiation")
    if os.path.isdir(irrad_root):
        for name in sorted(os.listdir(irrad_root)):
            path = os.path.join(irrad_root, name)
            if os.path.isdir(path):
                stages.append((path, "radiating"))

    post = os.path.join(base, "Postirradiation")
    if os.path.isdir(post):
        stages.append((post, "post"))

    return stages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FITS conversion, indexing and radiation analysis"
    )
    parser.add_argument("basepath", help="Path to dataset root")
    parser.add_argument(
        "radiation_log",
        help="Path to radiationLogCompleto.csv",
    )
    parser.add_argument("output_dir", help="Directory for analysis results")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=["pre", "radiating", "post"],
        help="Stages to analyse (pre, radiating, post)",
    )
    args = parser.parse_args()

    stages = _gather_stages(args.basepath)
    index_frames = []
    for path, stage in stages:
        bias = os.path.join(path, "Bias")
        dark = os.path.join(path, "Darks")
        flat = os.path.join(path, "Flats")

        raw_to_fits.convert_many(bias, dark, flat, search_depth=6)
        tmp_csv = os.path.join(path, "index.csv")
        index_dataset.index_sections(bias, dark, flat, tmp_csv, stage=stage, search_depth=6)
        if os.path.isfile(tmp_csv):
            index_frames.append(pd.read_csv(tmp_csv))

    if not index_frames:
        print("No data found for irradiation structure")
        return

    df = pd.concat(index_frames, ignore_index=True)
    index_csv = os.path.join(args.basepath, "index.csv")
    df.to_csv(index_csv, index=False)

    radiation_analysis.main(index_csv, args.radiation_log, args.output_dir, args.stages)


if __name__ == "__main__":
    main()
