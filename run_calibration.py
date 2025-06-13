#!/usr/bin/env python
"""Run raw_to_fits, index_dataset and process_index for different dataset structures."""

from __future__ import annotations

import argparse
import os
import pandas as pd

from utils import raw_to_fits, index_dataset
import process_index


def _run_standard(base: str, output_dir: str) -> None:
    bias = os.path.join(base, "TestSection1")
    dark = os.path.join(base, "TestSection2")
    flat = os.path.join(base, "TestSection3")

    raw_to_fits.convert_many(bias, dark, flat)
    index_csv = os.path.join(base, "index.csv")
    index_dataset.index_sections(bias, dark, flat, index_csv)
    process_index.main(index_csv, output_dir)


def _run_irradiation(base: str, output_dir: str) -> None:
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

    def _has_fits(root: str) -> bool:
        """Return ``True`` if ``root`` contains a ``fits/`` directory with files."""
        if not os.path.isdir(root):
            return False
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == "fits":
                if any(f.lower().endswith(".fits") for f in filenames):
                    return True
        return False

    index_frames = []
    for path, stage in stages:
        bias = os.path.join(path, "Bias")
        dark = os.path.join(path, "Darks")
        flat = os.path.join(path, "Flats")

        skip_bias = _has_fits(bias)
        skip_dark = _has_fits(dark)
        skip_flat = _has_fits(flat)

        if not (skip_bias and skip_dark and skip_flat):
            raw_to_fits.convert_many(
                bias,
                dark,
                flat,
                search_depth=6,
                skip_bias=skip_bias,
                skip_dark=skip_dark,
                skip_flat=skip_flat,
            )
        tmp_csv = os.path.join(path, "index.csv")
        index_dataset.index_sections(bias, dark, flat, tmp_csv, stage=stage, search_depth=6)
        if os.path.isfile(tmp_csv):
            index_frames.append(pd.read_csv(tmp_csv))

    if not index_frames:
        print("No data found for irradiation structure")
        return

    combined = pd.concat(index_frames, ignore_index=True)
    index_csv = os.path.join(base, "index.csv")
    combined.to_csv(index_csv, index=False)
    process_index.main(index_csv, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete calibration workflow")
    parser.add_argument("structure", choices=["standard", "irradiation"], help="Dataset directory layout")
    parser.add_argument("basepath", help="Path to dataset root")
    parser.add_argument("output_dir", help="Directory for processed results")
    args = parser.parse_args()

    if args.structure == "standard":
        _run_standard(args.basepath, args.output_dir)
    else:
        _run_irradiation(args.basepath, args.output_dir)


if __name__ == "__main__":
    main()
