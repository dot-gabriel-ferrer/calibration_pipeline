# coding: utf-8
"""Index FITS files from the three calibration sections.

The script scans the ``TestSection1`` (bias), ``TestSection2`` (dark) and
``TestSection3`` (flat) directories and generates a CSV summary of all FITS
files found.  Optionally the radiation stage and vacuum state can be
specified on the command line.  If not provided, they are inferred from the
directory names.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
from typing import Iterable, List, Optional

import numpy as np
from astropy.io import fits

from .raw_to_fits import gather_attempts

logger = logging.getLogger(__name__)


def _infer_stage(path: str) -> Optional[str]:
    for part in path.split(os.sep):
        low = part.lower()
        if "pre" in low:
            return "pre"
        if "during" in low:
            return "during"
        if "post" in low:
            return "post"
    return None


def _infer_vacuum(path: str) -> Optional[str]:
    for part in path.split(os.sep):
        low = part.lower()
        if "vac" in low:
            return "vacuum"
        if "atm" in low or "air" in low:
            return "air"
    return None


def _process_fits(path: str, caltype: str, stage: Optional[str], vacuum: Optional[str]) -> dict:
    with fits.open(path, mode="update") as hdul:
        data = hdul[0].data
        header = hdul[0].header
        temp = header.get("TEMP")
        if temp is None:
            temp = header.get("EQTEMP")
        zero_frac = float(np.count_nonzero(data == 0)) / data.size
        bad = zero_frac > 0.01
        if bad:
            header["BADFITS"] = True
            hdul.flush()
    return {
        "PATH": path,
        "CALTYPE": caltype,
        "STAGE": stage,
        "VACUUM": vacuum,
        "TEMP": temp,
        "ZEROFRACTION": zero_frac,
        "BADFITS": bad,
    }


def index_sections(
    bias_root: str,
    dark_root: str,
    flat_root: str,
    output_csv: str,
    *,
    stage: Optional[str] = None,
    vacuum: Optional[str] = None,
    search_depth: int = 6,
) -> None:
    rows: List[dict] = []
    datasets = [
        (bias_root, "BIAS"),
        (dark_root, "DARK"),
        (flat_root, "FLAT"),
    ]

    for root, cal in datasets:
        if not os.path.isdir(root):
            logger.warning("Dataset path %s does not exist", root)
            continue
        ds_stage = stage or _infer_stage(root)
        ds_vacuum = vacuum or _infer_vacuum(root)
        attempts = gather_attempts(root, max_depth=search_depth)
        if not attempts:
            attempts = [root]
        for attempt in attempts:
            fits_dir = os.path.join(attempt, "fits")
            patterns = [os.path.join(fits_dir, "*.fits"), os.path.join(attempt, "*.fits")]
            for pattern in patterns:
                for fpath in sorted(glob.glob(pattern)):
                    rows.append(_process_fits(fpath, cal, ds_stage, ds_vacuum))

    if not rows:
        logger.warning("No FITS files found")
        return

    fieldnames = ["PATH", "CALTYPE", "STAGE", "VACUUM", "TEMP", "ZEROFRACTION", "BADFITS"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote index to %s", output_csv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index FITS calibration dataset")
    parser.add_argument("bias_section", help="Path to TestSection1 (BIAS)")
    parser.add_argument("dark_section", help="Path to TestSection2 (DARK)")
    parser.add_argument("flat_section", help="Path to TestSection3 (FLAT)")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("--stage", choices=["pre", "during", "post"], help="Radiation stage")
    parser.add_argument("--vacuum", help="Vacuum state")
    parser.add_argument("--search-depth", type=int, default=6, help="Maximum directory depth when searching for attempts")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    index_sections(
        args.bias_section,
        args.dark_section,
        args.flat_section,
        args.output_csv,
        stage=args.stage,
        vacuum=args.vacuum,
        search_depth=args.search_depth,
    )


if __name__ == "__main__":
    main()
