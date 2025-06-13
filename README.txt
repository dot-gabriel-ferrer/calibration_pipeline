This README summarises the radiation dose analysis utilities.

Noise estimation
----------------
The `dose_analysis.py` script computes dynamic range and noise for each
irradiation dose. The noise in ADU is obtained from the quadratic sum of the
per-dose bias and dark standard deviations:

```
noise = sqrt(bias_std**2 + dark_std**2)
```

It is also reported as an equivalent magnitude uncertainty:

```
noise_mag = 1.0857 * noise / dynamic_range_16
```

where `dynamic_range_16` is the 16‑bit range (`65536 - bias_mean - dark_mean`).
Both values are stored in `dynamic_range.npz` alongside the dynamic range and
per-dose means.

Dynamic range reduction plots
-----------------------------
`dynamic_range_vs_dose_16.png` and `dynamic_range_vs_dose_12.png` show how the
available range decreases with dose for 16‑bit and 12‑bit readout
respectively.  Each figure includes a subplot with the percentage reduction
relative to the corresponding ideal range (65536 ADU or 4096 ADU).  A shaded
region illustrates the one sigma uncertainty of the dynamic range.

Slope of the base level increase
--------------------------------
`dose_analysis.py` also fits a linear trend of mean bias/dark level versus dose.
The slopes and intercepts are written to `base_level_trend.csv` and the fitted
lines are shown in `base_level_trend_bias.png` and
`base_level_trend_dark.png`. The slope column indicates how many ADU per kRad
the detector baseline rises during irradiation.

Saved `.npz` files
------------------
Several helper arrays are stored in the analysis directory:

* `dynamic_range.npz` – noise and dynamic range values from
  `_dynamic_range_analysis`.
* `stage_base_diff.npz` – baseline differences between irradiation stages from
  `_stage_base_level_diff`.

Load these files with `numpy.load` for custom plotting or further processing:

```python
arr = np.load('analysis/dynamic_range.npz')
print(arr['dose'], arr['noise_adu'])
```

The arrays correspond to the plots produced by the script and can be reused to
compare multiple datasets without re-running the full analysis.

