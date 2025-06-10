# Calibration Pipelines

This repository collects several pipelines to generate calibration reference frames for a CMOS-based imaging system. Each pipeline can be executed independently through its own command line wrapper.

## bias_pipeline
A temperature-dependent bias modeling pipeline. It groups bias frames by temperature, fits a per-pixel linear model (`bias = a + b*T`) and can generate synthetic bias frames. Details and command line options are documented in the [bias pipeline README](bias_pipeline/bias_pipeline/README.md).

## dark_pipeline
Builds a dark current model that depends on temperature and exposure time. It supports 1‑D and 2‑D models, synthetic dark generation and evaluation utilities. See the [dark pipeline README](dark_pipeline/README.md) for full instructions.

## flat_pipeline
Processes flat-field frames. Steps include optional dark subtraction, normalization, vignetting correction, master flat creation and flat model fitting. Modes are selected via `run_pipeline.py`.

## Analysis scripts
The repository also includes a couple of standalone scripts useful for data exploration:

- **`structure_analysis.py`** – inspects flat frames using `ObservationManager`.
  It can build bias/dark maps, optionally reduce the flats and plot trends over
  time or temperature in order to detect new structures.
- **`dark_pipeline/steps/outgasing_destruction_analysis.py`** – analyses flats
  for long‑term detector degradation. It normalises frames, searches for outlier
  pixels and generates summary plots. See the script for additional options.

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

# Outgasing/destruction analysis
python dark_pipeline/dark_pipeline/steps/outgasing_destruction_analysis.py \
    path/to/FITS --calibrate --output_dir analysis_results
```

## Raw to FITS Conversion

The `utils.raw_to_fits` module converts the raw camera frames of the calibration
datasets into FITS format. It expects the paths to the three dataset roots:
`TestSection1` (bias), `TestSection2` (dark) and `TestSection3` (flat).

In the bias section the tool scans directories named `T<temp>` and, inside each
of them, every `attempt<n>` folder.  Dark and flat datasets may include several
intermediate folders (e.g. `20Frames/<exptime>s/` or
`ContinuousFrames/T<temp>/<exptime>s`) before reaching the attempts.  The
converter now searches up to six directory levels by default so deeper
structures are handled automatically.  Each attempt must contain
`configFile.txt`, `temperatureLog.csv` and a `frames/` directory with the raw
files.  All such structures are handled automatically.


Run the conversion with:

```bash
python -m utils.raw_to_fits path/to/TestSection1 path/to/TestSection2 path/to/TestSection3
```

Add the `--verbose` option to print warnings about missing metadata.  The search
depth for dark and flat attempts can be changed with `--search-depth`, e.g.

```bash
python -m utils.raw_to_fits path/to/TestSection1 path/to/TestSection2 path/to/TestSection3 --verbose --search-depth 6
```

If the script reports no attempts found, the directory tree might be deeper than the default.
Re-run with a larger `--search-depth` (e.g. `--search-depth 8`).

Use `--skip-bias`, `--skip-dark` or `--skip-flat` to ignore a dataset section if
you only wish to convert a subset of the data.

For each attempt a `fits/` directory is created alongside `frames/` containing
the generated FITS files. All columns present in `temperatureLog.csv` are
written into the FITS header using short keywords (for instance
`FrameNum` → `FRAMENUM`, `ExpTime` → `EXPTIME`).  Certain columns are mapped
to more concise names: `ExpGain` → `GAIN`, `Temperature` → `TEMP`,
`InitialTemp` → `TEMP_0` and `ExtTemperature` → `EQTEMP`. Exposure time values are
converted from microseconds to seconds.  If the raw filename encodes the
exposure time (e.g. `exp0.1s` or `exp_1.2e-05s`) it is only used when the CSV
does not provide one. Any temperature indicated in the filename is stored under
`FILETEMP`.
If the CSV uses alternative column names for the detector temperature (e.g.
`CCDTemp`, `CCDTemperature` or `ChipTemp`), they are automatically normalised to
the header keyword `TEMP` so that downstream scripts can rely on a consistent
name.

After the conversion finishes a `fits_index.csv` file is written in the common
parent directory of the three sections. Each row lists the path to a generated
FITS file along with basic metadata (frame number, exposure time, temperatures
and calibration type).



For more options refer to the READMEs within each submodule.


