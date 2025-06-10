# coding: utf-8
"""Utility to convert raw camera frames into FITS files."""

import os
import csv
import glob
from typing import Dict, Iterable, Optional

import numpy as np
from astropy.io import fits


def read_config(path: str) -> Dict[str, str]:
    """Parse a simple key=value configuration file."""
    cfg: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def read_temperature_log(path: str) -> Dict[int, float]:
    """Read a temperatureLog.csv with frame index and temperature."""
    temps: Dict[int, float] = {}
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                frame = int(row[0])
                temp = float(row[1])
            except ValueError:
                # skip header or malformed rows
                continue
            temps[frame] = temp
    return temps


def _parse_dimensions(cfg: Dict[str, str]) -> tuple[int, int, int]:
    height = int(cfg.get("HEIGHT", cfg.get("height", 0)))
    width = int(cfg.get("WIDTH", cfg.get("width", 0)))
    bit_depth = int(cfg.get("BIT_DEPTH", cfg.get("bit_depth", 16)))
    if not height or not width:
        raise ValueError("Image dimensions missing in config file")
    return height, width, bit_depth


def _open_raw(path: str, height: int, width: int, dtype: np.dtype) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape((height, width))


def convert_section(section_path: str, raw_subdir: str = "frames") -> list[str]:
    """Convert all .raw files inside ``section_path`` to FITS files.

    Parameters
    ----------
    section_path: str
        Path to a section directory containing ``configFile.txt`` and
        ``temperatureLog.csv``.
    raw_subdir: str, optional
        Name of the sub-directory containing ``.raw`` files. If the directory
        does not exist, ``section_path`` is searched directly.

    Returns
    -------
    list[str]
        Paths to the generated FITS files.
    """
    config_path = os.path.join(section_path, "configFile.txt")
    temp_log_path = os.path.join(section_path, "temperatureLog.csv")
    raw_dir = os.path.join(section_path, raw_subdir)
    if not os.path.isdir(raw_dir):
        raw_dir = section_path

    out_dir = os.path.join(section_path, "fits")
    os.makedirs(out_dir, exist_ok=True)

    cfg = read_config(config_path)
    temps = read_temperature_log(temp_log_path)
    height, width, bit_depth = _parse_dimensions(cfg)

    dtype = np.uint16 if bit_depth > 8 else np.uint8

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.raw")))
    fits_paths = []
    for idx, raw_file in enumerate(raw_files):
        data = _open_raw(raw_file, height, width, dtype)
        header = fits.Header()
        for k, v in cfg.items():
            header[k.upper()] = v
        if idx in temps:
            header["CCD_TEMP"] = temps[idx]
        hdul = fits.HDUList([fits.PrimaryHDU(data, header=header)])
        out_name = os.path.splitext(os.path.basename(raw_file))[0] + ".fits"
        out_path = os.path.join(out_dir, out_name)
        hdul.writeto(out_path, overwrite=True)
        fits_paths.append(out_path)
    return fits_paths


def convert_many(sections: Iterable[str]) -> None:
    for sec in sections:
        print(f"Converting section {sec} ...")
        convert_section(sec)


def main(argv: Optional[Iterable[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw frames to FITS format using configuration files",
    )
    parser.add_argument(
        "sections",
        nargs="+",
        metavar="SECTION",
        help="Path(s) to acquisition sections",
    )
    args = parser.parse_args(argv)
    convert_many(args.sections)


if __name__ == "__main__":
    main()
