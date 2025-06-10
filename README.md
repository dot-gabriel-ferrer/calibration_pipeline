# Calibration Pipelines

This repository collects several pipelines to generate calibration reference frames for a CMOS-based imaging system. Each pipeline can be executed independently through its own command line wrapper.

## bias_pipeline
A temperature-dependent bias modeling pipeline. It groups bias frames by temperature, fits a per-pixel linear model (`bias = a + b*T`) and can generate synthetic bias frames. Details and command line options are documented in the [bias pipeline README](bias_pipeline/bias_pipeline/README.md).

## dark_pipeline
Builds a dark current model that depends on temperature and exposure time. It supports 1‑D and 2‑D models, synthetic dark generation and evaluation utilities. See the [dark pipeline README](dark_pipeline/README.md) for full instructions.

## flat_pipeline
Processes flat-field frames. Steps include optional dark subtraction, normalization, vignetting correction, master flat creation and flat model fitting. Modes are selected via `run_pipeline.py`.

## structure_analysis.py
Standalone example script for analysing flat frames. It loads bias and dark data with `ObservationManager`, generates bias/dark maps, optionally reduces flats and visualises temporal or temperature trends to spot new structures.

## Installation

Install the required Python packages using the repository wide `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Quick start
```bash
# Bias model
python bias_pipeline/bias_pipeline/run_pipeline.py --basepath path/to/bias --output-dir bias_results

# Dark model
python dark_pipeline/run_pipeline.py --mode full --basepath path/to/darks --output_dir dark_results

# Flat processing
python flat_pipeline/run_pipeline.py --mode full --basepath path/to/flats --output-dir flat_results

# Structure analysis example
python structure_analysis.py path/to/calibration [--reduce]
```

## Raw to FITS Conversion

The `utils.raw_to_fits` module converts the raw camera frames of the calibration
datasets into FITS format. It expects the paths to the three dataset roots:
`TestSection1` (bias), `TestSection2` (dark) and `TestSection3` (flat).

In the bias and dark sections the tool scans directories named `T<temp>` and,
inside each of them, every `attempt<n>` folder. Each attempt must contain
`configFile.txt`, `temperatureLog.csv` and a `frames/` directory with the raw
files.  The flat section may include an extra level (e.g. `20frames/`) before the
`T<temp>` folders, which are processed in the same way.

Run the conversion with:

```bash
python -m utils.raw_to_fits path/to/TestSection1 path/to/TestSection2 path/to/TestSection3
```

For each attempt a `fits/` directory is created alongside `frames/` containing
the generated FITS files. All columns present in `temperatureLog.csv` are
written into the FITS header using short keywords (for instance
`FrameNum` → `FRAMENUM`, `ExpTime` → `EXPTIME`).  Exposure time values are
converted from microseconds to seconds.  If the raw filename encodes the
exposure time (e.g. `exp0.1s` or `exp_1.2e-05s`) it is only used when the CSV
does not provide one. Any temperature indicated in the filename is stored under
`FILETEMP`.

For more options refer to the READMEs within each submodule.


