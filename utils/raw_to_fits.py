# coding: utf-8
"""Utility to convert raw camera frames into FITS files."""

import os
import csv
import glob
import re
from typing import Dict, Iterable, Optional, List, Tuple

DEFAULT_HEIGHT = 2048
DEFAULT_WIDTH = 2048
DEFAULT_BIT_DEPTH = 16

import numpy as np
from astropy.io import fits


def read_config(path: str) -> Dict[str, str]:
    """Parse a configuration file using ``key=value`` or ``key: value`` syntax."""

    cfg: Dict[str, str] = {}
    if not os.path.isfile(path):
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            elif ":" in line:
                k, v = line.split(":", 1)
            else:
                continue
            cfg[k.strip()] = v.strip()

    return cfg


def read_temperature_log(path: str) -> Dict[int, float]:
    """Read ``temperatureLog.csv`` and return a mapping ``frame -> temperature``."""

    temps: Dict[int, float] = {}
    if not os.path.isfile(path):
        return temps

    with open(path, newline="") as csvfile:
        # Try to autodetect delimiter and field names
        sample = csvfile.read(1024)
        csvfile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        csvfile.seek(0)
        reader = csv.DictReader(csvfile, dialect=dialect)

        if "Temperature" in reader.fieldnames:
            for row in reader:
                try:
                    frame = int(row.get("FrameNum", row.get("frame", row[reader.fieldnames[0]])))
                    temp = float(row["Temperature"])
                except (ValueError, KeyError):
                    continue
                temps[frame] = temp
        else:
            # Fallback: use raw reader and assume temperature is the 7th column
            csvfile.seek(0)
            simple_reader = csv.reader(csvfile, delimiter=dialect.delimiter)
            for row in simple_reader:
                if not row or len(row) < 7:
                    continue
                try:
                    frame = int(row[0])
                    temp = float(row[6])
                except ValueError:
                    continue
                temps[frame] = temp

    return temps


def _parse_dimensions(cfg: Dict[str, str]) -> tuple[int, int, int]:
    """Return image ``height``, ``width`` and ``bit_depth`` from config.

    Values fall back to defaults if not present.
    """

    height = int(cfg.get("HEIGHT", cfg.get("height", DEFAULT_HEIGHT)))
    width = int(cfg.get("WIDTH", cfg.get("width", DEFAULT_WIDTH)))
    bit_depth = int(cfg.get("BIT_DEPTH", cfg.get("bit_depth", DEFAULT_BIT_DEPTH)))
    return height, width, bit_depth


def _open_raw(path: str, height: int, width: int, dtype: np.dtype) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape((height, width))


def parse_filename_metadata(name: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract exposure time and temperature from a raw filename.

    Examples of supported patterns::

        BiasT0_exp0.012sAt0f1.raw
        exp_1.2e-05s_frame0.raw

    Parameters
    ----------
    name : str
        Base filename of the raw frame.

    Returns
    -------
    tuple
        ``(exptime_seconds, temperature)``. Values are ``None`` if not found.
    """

    exp_match = re.search(r"exp[_]?([0-9.+\-eE]+)s", name)
    exptime = None
    if exp_match:
        try:
            exptime = float(exp_match.group(1))
        except ValueError:
            exptime = None

    temp_match = re.search(r"T(-?[0-9]+(?:\.[0-9]+)?)", name)
    temp = None
    if temp_match:
        try:
            temp = float(temp_match.group(1))
        except ValueError:
            temp = None

    return exptime, temp


def convert_attempt(attempt_path: str, calibration: str, raw_subdir: str = "frames") -> List[str]:
    """Convert all ``.raw`` files inside an attempt directory into FITS files.

    The resulting FITS files are stored in a ``fits/`` subdirectory within the
    attempt. Basic configuration values and per-frame temperature information are
    written to the FITS header.

    Parameters
    ----------
    attempt_path: str
        Path to the attempt directory containing ``configFile.txt`` and
        ``temperatureLog.csv``.
    calibration: str
        Calibration type (``BIAS``, ``DARK`` or ``FLAT``) stored in the header.
    raw_subdir: str, optional
        Name of the sub-directory containing ``.raw`` files. If the directory
        does not exist, ``attempt_path`` is searched directly.

    Returns
    -------
    list[str]
        Paths to the generated FITS files.
    """
    config_path = os.path.join(attempt_path, "configFile.txt")
    if not os.path.isfile(config_path):
        config_path = os.path.join(attempt_path, "config.txt")

    temp_log_path = os.path.join(attempt_path, "temperatureLog.csv")

    raw_dir = os.path.join(attempt_path, raw_subdir)
    if not os.path.isdir(raw_dir):
        raw_dir = attempt_path

    out_dir = os.path.join(attempt_path, "fits")
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
        header["CALTYPE"] = calibration
        for k, v in cfg.items():
            header[k.upper()[:8]] = v
        if idx in temps:
            header["CCD_TEMP"] = temps[idx]
            header["TEMP"] = temps[idx]

        fname = os.path.basename(raw_file)
        exptime, fname_temp = parse_filename_metadata(fname)
        if exptime is not None:
            header["EXPTIME"] = exptime
        if fname_temp is not None:
            header.setdefault("FILETEMP", fname_temp)
        hdul = fits.HDUList([fits.PrimaryHDU(data, header=header)])
        out_name = os.path.splitext(os.path.basename(raw_file))[0] + ".fits"
        out_path = os.path.join(out_dir, out_name)
        hdul.writeto(out_path, overwrite=True)
        fits_paths.append(out_path)
    return fits_paths


def _find_attempts_in_temperature_dir(t_dir: str) -> List[str]:
    attempts: List[str] = []
    for entry in os.scandir(t_dir):
        if entry.is_dir() and entry.name.lower().startswith("attempt"):
            attempts.append(entry.path)
    return attempts


def gather_attempts(root: str, nested: bool = False) -> List[str]:
    """Collect attempt directories in ``root``.

    If ``nested`` is ``True`` (used for FLAT datasets), one level of subfolder is
    searched before looking for ``T<temp>`` directories.
    """

    attempt_dirs: List[str] = []

    def process_parent(parent: str):
        for entry in os.scandir(parent):
            if entry.is_dir() and re.match(r"^T-?\d+", entry.name):
                attempt_dirs.extend(_find_attempts_in_temperature_dir(entry.path))

    if nested:
        for sub in os.scandir(root):
            if sub.is_dir() and re.match(r"^T-?\d+", sub.name):
                process_parent(sub.path)
            elif sub.is_dir():
                for tdir in os.scandir(sub.path):
                    if tdir.is_dir() and re.match(r"^T-?\d+", tdir.name):
                        attempt_dirs.extend(_find_attempts_in_temperature_dir(tdir.path))
    else:
        process_parent(root)

    return attempt_dirs


def convert_many(bias_root: str, dark_root: str, flat_root: str) -> None:
    datasets = [
        (bias_root, "BIAS", False),
        (dark_root, "DARK", False),
        (flat_root, "FLAT", True),
    ]

    for root, caltype, nested in datasets:
        print(f"Processing {caltype} dataset at {root}...")
        for attempt in gather_attempts(root, nested=nested):
            print(f"  Converting attempt {attempt} ...")
            convert_attempt(attempt, calibration=caltype)


def main(argv: Optional[Iterable[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert raw frames from bias, dark and flat datasets to FITS."
        )
    )
    parser.add_argument("bias_section", help="Path to TestSection1 (BIAS)")
    parser.add_argument("dark_section", help="Path to TestSection2 (DARK)")
    parser.add_argument("flat_section", help="Path to TestSection3 (FLAT)")

    args = parser.parse_args(argv)

    convert_many(args.bias_section, args.dark_section, args.flat_section)


if __name__ == "__main__":
    main()
