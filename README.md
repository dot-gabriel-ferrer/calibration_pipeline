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
- **`operation_analysis.py`** – analyses radiation exposure sequences. It now
  correlates radiation logs with per-frame outlier counts and produces
  `rad_vs_outliers.png` for basic sensor behaviour assessment. The
  linear fit on this plot now gracefully falls back to a constant line
  if a polynomial fit cannot be computed.
- **`dose_analysis.py`** – groups calibration frames by radiation dose,
  fits a dose-response for dark current at each exposure time and now also
  fits linear base level trends for bias and dark frames. The coefficients are
  written to `analysis/base_level_trend.csv` alongside the corresponding plots.
  It also compares the first/last irradiation base levels with the pre/post
  values, saving the differences versus dose in `analysis/stage_base_diff.npz`
  and the accompanying figures. Plots of mean signal, photometric precision
  and magnitude/ADU error versus dose are stored in `plots/` together with
  matching `.npz` files (e.g. `mag_err_vs_dose.npz`) containing the plotted
  arrays.

The stored differences indicate how much the detector baseline shifts when
irradiation begins and how much of that shift remains once the source is turned
off. Positive values mean the radiating stage has a higher mean level than the
reference stage.

## Installation

Install the required Python packages, including `pandas`, using the repository wide `requirements.txt` file:

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



## Dataset Indexing

The `utils.index_dataset` module scans the converted FITS files and creates a CSV
summary. Provide the paths to the dataset sections and the output CSV location.
The `--bias`, `--dark` and `--flat` options may be specified multiple times to
index several datasets at once.  Each of these options **must** point directly to
the corresponding ``Bias``/``Darks``/``Flats`` directory (e.g. ``TestSection1``,
``TestSection2`` and ``TestSection3``).
Supplying a higher-level dataset root will cause dark or flat frames to be
indexed unintentionally under the bias section. If `--stage` or `--vacuum` are
omitted they are automatically inferred from the directory names.

Incorrect usage (pointing ``--bias`` to the dataset root):

```bash
python -m utils.index_dataset --bias path/to/dataset index.csv
```

Correct usage (pointing to the calibration directories):

```bash
python -m utils.index_dataset \
    --bias path/to/dataset/Bias \
    --dark path/to/dataset/Darks \
    --flat path/to/dataset/Flats \
    index.csv
```

You may also let the script discover the calibration folders automatically by
passing the dataset root with `--discover`. It searches for `Bias`, `Darks` and
`Flats` directories (case insensitive) up to the configured `--search-depth` and
assigns the corresponding `CALTYPE` automatically.

```bash
python -m utils.index_dataset --discover path/to/dataset index.csv
```

The resulting CSV contains the following columns:
`PATH`, `CALTYPE`, `STAGE`, `VACUUM`, `TEMP`, `ZEROFRACTION` and `BADFITS`.
Each row corresponds to a FITS file and the script tags files with
`BADFITS=True` when more than 1% of their pixels are zero.

## Index analysis

Once you have created ``index.csv`` you can run ``process_index.py`` to
automatically build master calibration frames, compute statistics for every
FITS file and generate simple trend plots.

```bash
python process_index.py path/to/index.csv output_dir/
```

The output directory will contain the generated masters, a ``frame_stats.csv``
file with per-frame metrics and several PNG figures inside ``plots/`` and
``comparisons/``.

## Automated workflow with ``run_calibration.py``

The ``run_calibration.py`` script ties together dataset conversion,
indexing and processing.  It calls ``utils.raw_to_fits``,
``utils.index_dataset`` and ``process_index.py`` for a complete
end-to-end run.  The first argument selects the dataset layout:
``standard`` for the usual ``TestSection1``/``2``/``3`` structure or
``irradiation`` for radiation test campaigns with ``Preirradiation``
and ``Postirradiation`` folders.

Example for a standard dataset:

```bash
python run_calibration.py standard path/to/dataset output_dir/
```

For irradiation campaigns:

```bash
python run_calibration.py irradiation path/to/irrad_dataset output_dir/
```
In this mode the script inspects every ``Bias``, ``Darks`` and ``Flats``
folder. Raw-to-FITS conversion is only executed when no ``fits/``
directory with FITS files is present so previously converted data is not
processed again.

The script writes an ``index.csv`` file in the dataset root and fills the
given output directory with the generated masters, the
``frame_stats.csv`` summary and all plots and comparisons described in the
previous section.

## Irradiation workflow

For radiation campaigns, `run_radiation.py` automates FITS conversion,
dataset indexing and the analysis performed by `radiation_analysis.py`.
It expects the same directory structure used by `run_calibration.py`
with `Preirradiation`, `Irradiation` and `Postirradiation` folders and a
`radiationLogCompleto.csv` file.

```bash
python run_radiation.py path/to/irrad_dataset path/to/radiationLogCompleto.csv output_dir/
```

Select specific stages with `--stages pre radiating post` (defaults to all).
The script writes `index.csv` in the dataset root and places all analysis
outputs inside the chosen output directory. Difference heatmaps created during
the analysis now save the underlying arrays alongside each PNG as `.npz`
files for further inspection.
