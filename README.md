# Calibration Pipeline Utilities

This repository contains pipelines and utilities for processing calibration frames.

## Raw to FITS Conversion

The `utils` package provides a helper to convert `.raw` camera frames into FITS
files.  Each acquisition section should contain:

- `configFile.txt` – configuration parameters (e.g. `HEIGHT`, `WIDTH`,
  `BIT_DEPTH`).
- `temperatureLog.csv` – CSV with two columns: frame index and sensor
  temperature.
- A folder with the `.raw` frames (defaults to `frames/`).

To convert one or more sections run:

```bash
python -m utils.raw_to_fits <TestSection1> <TestSection2> <TestSection3>
```

A `fits/` subdirectory will be created inside each section containing the
resulting FITS files with the configuration values and the corresponding frame
temperature written into the header.
