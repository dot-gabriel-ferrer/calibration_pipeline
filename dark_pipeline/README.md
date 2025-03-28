
# Dark Model Pipeline for PhotSat Calibration

Author: Elías Gabriel Ferrer Jorge

Project: PhotSat Synthetic Dark Modeling

---

## 🌟 Overview

This pipeline builds a **temperature and exposure time dependent dark current model** for CMOS-based space instrumentation. The approach is modular, reproducible, and optimized for in-flight calibration datasets.

The pipeline:

* Loads long and short dark calibration exposures.
* Groups long darks by exposure time and temperature.
* Subtracts bias (from short darks) to isolate dark current.
* Fits a per-pixel non-linear model.
* Supports two dark current models:

### ✅ 2D Model (temperature and exposure dependent)

```
DC(T, t_exp) = A * t_exp^gamma * exp(B * (T - T_min))
```

### ✅ 1D Model (temperature dependent, fixed exposure)

```
DC(T) = A * exp(B * (T - T_min))
```

* Generates synthetic darks.
* Evaluates model accuracy (MAE, MAPE).

---

## 📂 Directory Structure

```
dark_pipeline/
├── main_pipeline.py              # Runs the full pipeline step-by-step
├── run_pipeline.py              # CLI entry point
├── steps/
│   ├── step0_create_dirs.py
│   ├── step1_load_data.py
│   ├── step2_subtract_bias.py
│   ├── step3_group_darks_by_temp_exp.py
│   ├── step4_mask_hotpixels.py
│   ├── step5_fit_dark_model.py
│   ├── step6_generate_synthetic_darks.py
│   ├── step7_evaluate_model.py
│   └── utils_scaling.py
```

---

## 🧪 Input Requirements

* FITS images from dark calibration campaigns.
* Required metadata in headers:
  * `'temperature'` [°C]
  * `'exposure'` [s]
* Images must be 16-bit scaled from original 12-bit (use `utils_scaling.py`).

---

## 🚀 How to Run the Pipeline

```bash
python run_pipeline.py \
  --output_dir path/to/save/outputs/ \
  --mode full \
  --basepath path/to/fits/
```

### Available Modes

| Mode                    | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| `full`                | Full pipeline: load, subtract, fit, evaluate                                     |
| `synthetic_only`      | Generate synthetic darks from existing model                                     |
| `evaluation_only`     | Evaluate synthetic vs real darks from model                                      |
| `fit_2d_model`        | Fit 2D model (T, t_exp dependent)                                                |
| `evaluate_2d_model`   | Evaluate 2D model using real master darks                                        |
| `synthetic_2d`        | Generate all synthetic 2D darks                                                  |
| `synthetic_2d_single` | Generate one synthetic 2D dark (requires `--temp_single`,`--exptime_single`) |

### Additional Arguments

| Flag                 | Description                                        |
| -------------------- | -------------------------------------------------- |
| `--model_dir`      | Path to model directory (for synthetic/eval modes) |
| `--temps`          | List of temperatures (e.g. '15.0 20.0 25.0')       |
| `--temp_single`    | Temperature for 2D single generation               |
| `--exptime_single` | Exposure time for 2D single generation             |

---

## ⚙️ Model Description

The pipeline supports two models for dark current estimation:

### 2D Model:

```
DC(T, t_exp) = A * t_exp^gamma * exp(B * (T - T_min))
```

Used when both temperature and exposure vary.

### 1D Model:

```
DC(T) = A * exp(B * (T - T_min))
```

Used when exposure is fixed or pre-corrected.

Model parameters (A, B, gamma) are fitted per pixel using non-linear optimization.

---

## ⚖️ Evaluation

The evaluation step compares synthetic vs. real darks:

* MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error) per pixel.
* Heatmaps for each temperature-exposure pair.
* Summary plots of average model performance.

---

## 📊 Output Example

```
output_dir/
├── dark_corrected_darks/
├── dark_masks_by_temp/
│   └── darks_TxxC_texpYYs.fits
├── dark_model/
│   ├── dark_A_map.fits
│   ├── dark_B_map.fits
│   ├── dark_gamma_map.fits
│   └── hot_pixel_mask.fits
└── evaluation/
    ├── mae_TxxC_texpYYs.png
    ├── mape_TxxC_texpYYs.png
    ├── synthetic_dark_TxxC_texpYYs.png
    └── evaluation_summary.png
```

---

## 📝 Notes

* All dark values are internally scaled from 16-bit to 12-bit units using `SCALE_FACTOR = 16.0`.
* Supports exclusion of hot pixels during model fitting.
* Compatible with separately generated bias models.

---

## 🔗 Integration

This pipeline is fully compatible with `master_controller.py` for coordinated execution with the `bias_pipeline`.

---
