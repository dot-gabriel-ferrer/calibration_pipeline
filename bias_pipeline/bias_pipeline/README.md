# Bias Model Pipeline – PhotSat Calibration

**Author:** Elías Gabriel Ferrer Jorge

**Project:** PhotSat – Synthetic Bias Modeling for CMOS Calibration

## Overview

This pipeline implements a temperature-dependent modeling of the electronic bias signal in space-based CMOS detectors. It follows a modular, step-based architecture and supports high-resolution master frame generation, pixel-wise linear regression modeling, and synthetic frame generation.

The pipeline is designed to:

* Organize and filter short-exposure  **bias calibration frames** .
* Group frames by sensor temperature and compute  **master bias frames** .
* Fit a **per-pixel linear model** to describe the temperature dependence:
  ```
  bias_ij(T) = a_ij + b_ij * T
  ```
* Generate synthetic bias frames at arbitrary temperatures.
* Evaluate model accuracy (per-pixel MAE, MAPE) and generate diagnostic plots.

---

## Pipeline Structure

```
bias_pipeline/
├── main_pipeline.py                  # Full step-by-step execution
├── run_pipeline.py                  # CLI wrapper for pipeline execution
├── generate_bias.py                 # Standalone utility to generate synthetic bias
├── steps/
│   ├── step0_create_dirs.py             # Create directory structure
│   ├── step1_load_data.py               # Load and filter bias FITS files
│   ├── step2_generate_master_bias_by_temp.py  # Master bias per temperature
│   ├── step3_fit_bias_model.py          # Pixel-wise linear regression
│   ├── step4_generate_synthetic_bias.py # Synthesize bias via model
│   ├── step5_evaluate_model.py          # Compute MAE, MAPE, generate plots
│   └── utils_scaling.py                 # 12-bit to 16-bit normalization
```

---

## Input Requirements

* **FITS images** from 12-bit CMOS detectors, rescaled to 16-bit.
* Valid FITS metadata headers including:
  * `TEMPERATURE` [°C]
  * `EXPOSURE` [s]
* Optional: hot pixel mask FITS file to exclude unstable pixels from model fitting.

---

## Running the Pipeline

### Full Execution

```bash
python run_pipeline.py \
  --basepath path/to/raw_fits/ \
  --output-dir path/to/results/ \
  --hot-pixel-mask path/to/hot_pixel_mask.fits \
  --generate-set \
  --save-eval-fits
```

### CLI Parameters

| Flag                 | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `--basepath`       | Directory containing raw FITS bias frames.                                   |
| `--output-dir`     | Output directory for model products, synthetic frames, and diagnostics.      |
| `--hot-pixel-mask` | (Optional) FITS mask for excluding hot pixels during regression.             |
| `--generate-set`   | If set, generates synthetic bias frames for all available temperatures.      |
| `--save-eval-fits` | If set, saves FITS files of synthetic bias, MAE, and MAPE during evaluation. |

---

## Generating a Single Synthetic Bias Frame

Use the standalone utility `generate_bias.py` to generate a synthetic frame at any temperature:

```bash
python generate_bias.py \
  --a-map path/to/bias_a_map.fits \
  --b-map path/to/bias_b_map.fits \
  --temperature 12.0 \
  --output path/to/output_dir/
```

* If `--output` is a  **directory** , the file is named `synthetic_bias_12.0C.fits`.
* Prints summary statistics (mean, std) of the generated frame.

---

## Output Structure

```
output_dir/
├── bias_grouped_by_temp/
│   └── master_bias_XX.XC.fits        # Real master frames grouped by temperature
├── bias_model/
│   ├── bias_a_map.fits               # Intercept map (a_ij)
│   └── bias_b_map.fits               # Slope map (b_ij)
├── bias_masks_by_temp/
│   └── synthetic_bias_XX.XC.fits     # Synthetic bias from model
└── evaluation/
    ├── real_bias_XX.XC.png
    ├── synthetic_bias_XX.XC.png
    ├── mae_XX.XC.png
    ├── mape_XX.XC.png
    ├── evaluation_summary.png
    └── (optional) synthetic_bias_XX.XC.fits, mae_XX.XC.fits, mape_XX.XC.fits
```

---

## Notes

* All pixel values are linearly rescaled from 12-bit to 16-bit via a constant scale factor.
* The bias model is fit independently for each pixel, using linear least-squares regression.
* MAE and MAPE plots visualize pixel-wise error across the sensor array and track accuracy as a function of temperature.

---

## Integration

This pipeline is designed for full compatibility with the `dark_pipeline` and other calibration modules in the PhotSat processing suite. It can be executed independently or invoked as part of a higher-level orchestration script.
