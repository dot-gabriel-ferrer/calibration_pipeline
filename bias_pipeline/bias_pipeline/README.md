# Bias Model Pipeline for PhotSat Calibration

Author: Elías Gabriel Ferrer Jorge

Project: PhotSat Synthetic Bias Modeling


## 🌟 Overview

This pipeline generates a **temperature-dependent bias model** for space-based CMOS observations. It follows a modular step-based structure, aligned with the dark current model pipeline. The goal is to:

* Load and organize short-exposure calibration frames (bias).
* Group frames by temperature and generate master biases.
* Fit a linear pixel-wise bias model of the form:

```
bias_ij(T) = a_ij + b_ij * T
```

* Generate synthetic bias frames.
* Evaluate the model performance and visualize errors (MAE, MAPE).

---

## 📂 Directory Structure

```
bias_pipeline/
├── main_pipeline.py              # Runs the full pipeline step-by-step
├── run_pipeline.py              # CLI to execute the full pipeline
├── generate_bias.py             # Utility to generate a single synthetic bias at any temperature
├── steps/
│   ├── step0_create_dirs.py
│   ├── step1_load_data.py
│   ├── step2_generate_master_bias_by_temp.py
│   ├── step3_fit_bias_model.py
│   ├── step4_generate_synthetic_bias.py
│   ├── step5_evaluate_model.py
│   └── utils_scaling.py         # Applies 12-bit to 16-bit rescaling for input data
```

---

## 🧪 Input Requirements

* FITS images in 16-bit format originally scaled from 12-bit CMOS (use `utils_scaling.py`).
* Metadata must include:
  * `'temperature'` [°C]
  * `'exposure'` [s]

---

## 🚀 How to Run the Pipeline

### 1. Execute Full Pipeline

```bash
python run_pipeline.py \
  --basepath path/to/raw_fits/ \
  --output-dir path/to/save_outputs/ \
  --hot-pixel-mask path/to/dark_model/hot_pixel_mask.fits \
  --generate-set \
  --save-eval-fits
```

### Arguments:

| Flag                 | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `--basepath`       | Path to the folder with FITS bias files               |
| `--output-dir`     | Where to save model, grouped data, evaluation         |
| `--hot-pixel-mask` | (Optional) mask to exclude hot pixels                 |
| `--generate-set`   | If set, generates synthetic bias for all temperatures |
| `--save-eval-fits` | If set, saves FITS files in evaluation step           |

---

## ⚙️ Generate a Synthetic Bias from Terminal

Use `generate_bias.py` to create a synthetic bias at any temperature:

```bash
python generate_bias.py \
  --a-map path/to/bias_a_map.fits \
  --b-map path/to/bias_b_map.fits \
  --temperature 12.0 \
  --output path/to/output_dir/
```

If you pass a directory as `--output`, the file will be saved as `synthetic_bias_12.0C.fits` inside it.

✅ It prints basic stats (mean and std) of the generated bias.

---

## 📊 Output Example

The pipeline will produce:

```
output_dir/
├── bias_grouped_by_temp/
│   └── master_bias_XX.XC.fits
├── bias_model/
│   ├── bias_a_map.fits
│   └── bias_b_map.fits
├── bias_masks_by_temp/
│   └── synthetic_bias_XX.XC.fits
└── evaluation/
    ├── mae_XX.XC.png
    ├── mape_XX.XC.png
    ├── synthetic_bias_XX.XC.png
    ├── real_bias_XX.XC.png
    └── evaluation_summary.png
```

---

## 📌 Notes

* All pixel values are scaled **from 16-bit back to 12-bit** using `SCALE_FACTOR = 16.0`.
* The model is linear per pixel and works well for most temperature ranges, but see `evaluation_summary.png` for performance diagnostics.

---

## 🔗 Integration

This pipeline is fully compatible with the `master_controller.py` for combined execution with the dark pipeline.

---
