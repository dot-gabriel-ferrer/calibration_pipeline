# coding: utf-8
"""Index FITS files from calibration datasets.

The script scans the ``TestSection1`` (bias), ``TestSection2`` (dark) and
``TestSection3`` (flat) directories of one or more datasets and generates a CSV
summary of all FITS files found.  Optionally the radiation stage and vacuum
state can be specified on the command line.  If not provided, they are inferred
from the directory names.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
from typing import Iterable, List, Optional, Sequence, Union

from tqdm import tqdm

import numpy as np
from astropy.io import fits

from .raw_to_fits import gather_attempts

logger = logging.getLogger(__name__)


def _infer_stage(path: str) -> Optional[str]:
    parts = path.split(os.sep)
    for part in reversed(parts):
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
    bias_root: Union[str, Sequence[str]],
    dark_root: Union[str, Sequence[str]],
    flat_root: Union[str, Sequence[str]],
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

    for roots, cal in datasets:
        if not roots:
            continue
        if isinstance(roots, str):
            roots = [roots]
        for root in roots:
            if not os.path.isdir(root):
                logger.warning("Dataset path %s does not exist", root)
                continue
            ds_stage = stage or _infer_stage(root)
            ds_vacuum = vacuum or _infer_vacuum(root)
            attempts = gather_attempts(root, max_depth=search_depth)
            if not attempts:
                attempts = [root]
            for attempt in tqdm(attempts, desc=f"{cal} attempts", ncols=80):
                fits_dir = os.path.join(attempt, "fits")
                patterns = [os.path.join(fits_dir, "*.fits"), os.path.join(attempt, "*.fits")]
                files: List[str] = []
                for pattern in patterns:
                    files.extend(sorted(glob.glob(pattern)))
                for fpath in tqdm(files, desc=os.path.basename(attempt), leave=False, ncols=80):
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
    parser.add_argument(
        "--bias",
        dest="bias_sections",
        action="append",
        help="Path to TestSection1 (BIAS). Can be used multiple times",
    )
    parser.add_argument(
        "--dark",
        dest="dark_sections",
        action="append",
        help="Path to TestSection2 (DARK). Can be used multiple times",
    )
    parser.add_argument(
        "--flat",
        dest="flat_sections",
        action="append",
        help="Path to TestSection3 (FLAT). Can be used multiple times",
    )
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("positional", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=["pre", "during", "post"], help="Radiation stage")
    parser.add_argument("--vacuum", help="Vacuum state")
    parser.add_argument("--search-depth", type=int, default=6, help="Maximum directory depth when searching for attempts")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    bias_sections = args.bias_sections or []
    dark_sections = args.dark_sections or []
    flat_sections = args.flat_sections or []

    if args.positional:
        if len(args.positional) >= 3:
            bias_sections.append(args.positional[0])
            dark_sections.append(args.positional[1])
            flat_sections.append(args.positional[2])
            args.positional = args.positional[3:]
        if args.positional:
            parser.error("Unexpected positional arguments: %s" % " ".join(args.positional))

    if not (bias_sections or dark_sections or flat_sections):
        parser.error("No dataset sections specified")

    index_sections(
        bias_sections,
        dark_sections,
        flat_sections,
        args.output_csv,
        stage=args.stage,
        vacuum=args.vacuum,
        search_depth=args.search_depth,
    )


if __name__ == "__main__":
    main()
