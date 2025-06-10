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


def load_csv_metadata(path: str) -> Dict[int, Dict[str, str]]:
    """Read ``temperatureLog.csv`` and return rows indexed by ``FrameNum``."""

    rows: Dict[int, Dict[str, str]] = {}
    if not os.path.isfile(path):
        return rows

    with open(path, newline="") as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        csvfile.seek(0)
        reader = csv.DictReader(csvfile, dialect=dialect)
        for row in reader:
            try:
                frame = int(row.get("FrameNum", row.get("frame", row[reader.fieldnames[0]])))
            except (ValueError, KeyError):
                continue
            rows[frame] = row

    return rows


def _parse_dimensions(cfg: Dict[str, str]) -> tuple[int, int, int]:
    """Return image ``height``, ``width`` and ``bit_depth`` from config.

    Values fall back to defaults if not present.
    """

    height = int(cfg.get("HEIGHT", cfg.get("height", DEFAULT_HEIGHT)))
    width = int(cfg.get("WIDTH", cfg.get("width", DEFAULT_WIDTH)))
    bit_depth = int(cfg.get("BIT_DEPTH", cfg.get("bit_depth", DEFAULT_BIT_DEPTH)))
    return height, width, bit_depth


def adapt_metadata_keys(row_metadata: Dict[str, str]) -> Dict[str, object]:
    """Map CSV columns to FITS header keywords and convert values."""

    mapping = {
        "FrameNum": "FRAMENUM",
        "TimeStamp": "TIMESTAMP",
        "ExtTemperature": "EXTTEMP",
        "ExpTime": "EXPTIME",
        "RealExpTime": "REXPTIME",
        "ExpGain": "EXPGAIN",
        "Temperature": "TEMP",
        "InitialTemp": "INITTMP",
        "DeltaTemperature": "DELTMP",
        "PowerCons": "POWCONS",
    }

    adapted: Dict[str, object] = {}
    for key, value in row_metadata.items():
        new_key = mapping.get(key, key.upper())
        if value is None:
            continue
        value = str(value).strip()
        if value == "":
            continue
        try:
            num = float(value)
            if np.isinf(num):
                adapted[new_key] = "INF"
            else:
                if key in ["ExpTime", "RealExpTime"]:
                    adapted[new_key] = num / 1e6
                else:
                    adapted[new_key] = num
        except ValueError:
            adapted[new_key] = value

    return adapted



def adapt_config_key(key: str) -> str:
    """Return a valid FITS header keyword for a config entry."""
    key = key.strip().upper().replace(" ", "_")
    key = re.sub(r"[^A-Z0-9_]+", "", key)
    return key[:8]


def parse_frame_number(name: str) -> Optional[int]:
    match = re.search(r"f(\d+)", name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


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


def convert_attempt(
    attempt_path: str,
    calibration: str,
    raw_subdir: str = "frames",
    index_rows: Optional[List[Dict[str, object]]] = None,
) -> List[str]:
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
    metadata = load_csv_metadata(temp_log_path) or {}

    height, width, bit_depth = _parse_dimensions(cfg)

    dtype = np.uint16 if bit_depth > 8 else np.uint8

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.raw")))
    fits_paths = []
    for idx, raw_file in enumerate(raw_files):
        data = _open_raw(raw_file, height, width, dtype)
        header = fits.Header()
        header["CALTYPE"] = calibration
        for k, v in cfg.items():
            header[adapt_config_key(k)] = v

        frame_num = parse_frame_number(os.path.basename(raw_file))
        row_meta = metadata.get(frame_num, {})
        if row_meta:
            adapted = adapt_metadata_keys(row_meta)
            for hk, hv in adapted.items():
                header[hk] = hv

        fname = os.path.basename(raw_file)
        exptime, fname_temp = parse_filename_metadata(fname)
        if exptime is not None and "EXPTIME" not in header:
            header["EXPTIME"] = exptime
        if fname_temp is not None:
            header.setdefault("FILETEMP", fname_temp)
        hdul = fits.HDUList([fits.PrimaryHDU(data, header=header)])
        out_name = os.path.splitext(os.path.basename(raw_file))[0] + ".fits"
        out_path = os.path.join(out_dir, out_name)
        hdul.writeto(out_path, overwrite=True)
        fits_paths.append(out_path)

        if index_rows is not None:
            row = {"PATH": out_path, "CALTYPE": calibration}
            for key in (
                "FRAMENUM",
                "EXPTIME",
                "REXPTIME",
                "TEMP",
                "FILETEMP",
                "EXTTEMP",
            ):
                if key in header:
                    row[key] = header[key]
            index_rows.append(row)
    return fits_paths


def gather_attempts(root: str, max_depth: int = 2) -> List[str]:
    """Return all attempt directories within ``root`` up to ``max_depth`` levels."""

    attempt_dirs: List[str] = []

    for dirpath, dirnames, _ in os.walk(root):
        rel_depth = os.path.relpath(dirpath, root).count(os.sep)
        if rel_depth > max_depth:
            continue
        for dname in dirnames:
            if dname.lower().startswith("attempt"):
                attempt_dirs.append(os.path.join(dirpath, dname))



    return attempt_dirs


def convert_many(bias_root: str, dark_root: str, flat_root: str) -> None:
    index_rows: List[Dict[str, object]] = []

    datasets = [
        # search depth 2 should be enough for bias (T<temp>/attempt<n>)
        (bias_root, "BIAS", 2),
        # dark and flat datasets may include additional folders like exposure time
        (dark_root, "DARK", 4),
        (flat_root, "FLAT", 4),
    ]

    for root, caltype, depth in datasets:
        print(f"Processing {caltype} dataset at {root}...")
        for attempt in gather_attempts(root, max_depth=depth):
            print(f"  Converting attempt {attempt} ...")
            convert_attempt(attempt, calibration=caltype, index_rows=index_rows)

    if index_rows:
        common_root = os.path.commonpath([bias_root, dark_root, flat_root])
        csv_path = os.path.join(common_root, "fits_index.csv")
        fieldnames = sorted({k for row in index_rows for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in index_rows:
                writer.writerow(row)
        print(f"Wrote index CSV to {csv_path}")



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
