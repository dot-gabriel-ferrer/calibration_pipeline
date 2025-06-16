#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`run_full_analysis_radiation`.

This script simply forwards its command line arguments to
:func:`run_full_analysis_radiation.run_pipeline` so that existing
documentation referencing ``run_full_analysis_pipeline.py`` keeps
working.
"""

from __future__ import annotations

import argparse
import logging

from run_full_analysis_radiation import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full analysis on irradiation FITS directories"
    )
    parser.add_argument("dataset_root", help="Directory containing <dose>kRads folders")
    parser.add_argument("output_dir", help="Directory for results")
    parser.add_argument(
        "--ignore-temp",
        action="store_true",
        help="Do not group frames by temperature",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_pipeline(
        args.dataset_root,
        args.output_dir,
        ignore_temp=args.ignore_temp,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
