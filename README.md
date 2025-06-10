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

For more options refer to the READMEs within each submodule.
