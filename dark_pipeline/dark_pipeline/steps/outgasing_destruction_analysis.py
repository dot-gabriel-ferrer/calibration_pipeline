#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UPDATED SCRIPT ADDRESSING NEW REQUIREMENTS:

 - CSV columns are fixed: each row is correctly delimited, so 'temp_labels' and 'times' won't break columns.
 - Time-based plots downsample tick labels if too many points.
 - Adds a scatter plot of outlier pixel positions for each temperature, sized by occurrence count.
 - Chooses the reference frame as the one with the SMALLEST timestamp in each group (by temperature or ignoring temperature).
 - Allows a custom output directory via --output_dir (defaults to 'outgasing_analysis').
 - Normalizes flats by max, storing them as FITS (not PNG).
 - Adds a sensor-plane outlier plot color-coded by classification.
 - Introduces an outline for deeper outlier analysis steps.

Usage Example:
    python this_script.py /path/to/FITS --calibrate --output_dir my_analysis_results
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
from collections import defaultdict

# Adjust as needed for your environment
try:
    from observation_manager.observation_manager import ObservationManager
except ImportError:
    print("Could not import ObservationManager. Ensure it's installed or on PYTHONPATH.")
    sys.exit(1)


##############################################################################
# HELPER: parse timestamp (raw string -> float), fallback if parse fails
##############################################################################
def parse_timestamp(time_str):
    """
    Attempt to parse the time_str as float for comparison.
    If it fails, return None (or large number).
    """
    try:
        return float(time_str)
    except:
        return None


##############################################################################
# 1) LOAD OBSERVATIONS
##############################################################################
def load_observations(basepath: str):
    manager = ObservationManager(base_path=basepath)
    manager.load_and_organize()

    # Filter darks (long exposures > 0.1)
    long_darks = manager.filter_files(
        category='CALIBRATION',
        subcat='darks',
        exp_min=0.1,
        ext_filter='fits'
    )
    # Filter bias (exposure <= 0.1)
    short_darks = manager.filter_files(
        category='CALIBRATION',
        subcat='bias',
        exp_max=0.1,
        ext_filter='fits'
    )

    return manager, long_darks, short_darks


##############################################################################
# 2) CHECK SUSPICIOUS BIAS & GENERATE BIAS MAPS
##############################################################################
def check_bias_frames(short_darks, zero_fraction_threshold=0.1, log_path='suspicious_bias.log'):
    good_bias = []
    suspicious_bias = []
    
    with open(log_path, 'w') as log_file:
        log_file.write("Suspicious bias frames (large zero-value areas):\n")

        for dark in tqdm(short_darks, desc="Checking Bias"):
            file_path = dark['original_path']
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)

            zero_fraction = np.mean(data == 0)
            if zero_fraction > zero_fraction_threshold:
                suspicious_bias.append(dark)
                log_file.write(f"{file_path} -> Zero fraction = {zero_fraction:.2%}\n")
            else:
                good_bias.append(dark)

    return good_bias, suspicious_bias


def generate_bias_maps(short_darks: list) -> dict:
    bias_groups = defaultdict(list)
    for dark in tqdm(short_darks, desc="Generating Bias Maps"):
        temp = dark.get('temperature', None)
        if temp is None:
            continue
        with fits.open(dark['original_path']) as hdul:
            data = hdul[0].data.astype(np.float32)
        bias_groups[temp].append(data)

    bias_map_by_temp = {}
    for temp, data_list in bias_groups.items():
        stack = np.stack(data_list, axis=0)
        bias_map_by_temp[temp] = np.mean(stack, axis=0)
    return bias_map_by_temp


##############################################################################
# 3) DARK MAPS (OPTIONAL)
##############################################################################
def generate_dark_maps(long_darks: list, temp_round=1.0, exp_round=0.1):
    dark_groups = defaultdict(list)
    for dark in tqdm(long_darks, desc="Generating Dark Maps"):
        temp = dark.get('temperature', None)
        exp = dark.get('exposure_time', None)
        if (temp is None) or (exp is None):
            continue

        t_r = round(temp, 1) if temp_round == 1.0 else round(temp, 2)
        e_r = round(exp, 1) if exp_round == 0.1 else round(exp, 2)

        with fits.open(dark['original_path']) as hdul:
            data = hdul[0].data.astype(np.float32)

        dark_groups[(t_r, e_r)].append(data)

    dark_map_by_temp_exp = {}
    for key, data_list in dark_groups.items():
        stack = np.stack(data_list, axis=0)
        dark_map_by_temp_exp[key] = np.mean(stack, axis=0)
    return dark_map_by_temp_exp


def find_closest_dark(temp, exp_time, dark_map_dict, temp_tolerance=1.0, exp_tolerance=0.1):
    if not dark_map_dict:
        return None
    candidates = []
    for (t_key, e_key), dark_data in dark_map_dict.items():
        diff_t = abs(t_key - round(temp, 1))
        diff_e = abs(e_key - round(exp_time, 1))
        if (diff_t <= temp_tolerance) and (diff_e <= exp_tolerance):
            candidates.append((diff_t + diff_e, dark_data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


##############################################################################
# 4) LOAD FLATS, Discard saturated, NORMALIZE by MAX, optionally subtract BIAS/DARK
##############################################################################
def load_flats(manager, bias_map_by_temp=None, dark_map_by_temp_exp=None,
               do_calibration=False, saturation_threshold=60000.0,
               output_folder="outgasing_analysis"):
    """
    - If do_calibration=True => subtract bias/dark first.
    - Then normalize each flat by its max => range ~ [0,1].
    - Save the result as a FITS in either 'flats_norm' or 'flats_norm_reduced' inside output_folder.
    """
    if do_calibration:
        norm_dir = os.path.join(output_folder, "flats_norm_reduced")
    else:
        norm_dir = os.path.join(output_folder, "flats_norm")
    os.makedirs(norm_dir, exist_ok=True)

    flats = manager.filter_files(
        category='CALIBRATION',
        subcat='flats',
        ext_filter='fits'
    )

    flats_data = []
    for flat_info in tqdm(flats, desc="Loading Flats"):
        file_path = flat_info['original_path']
        temp = flat_info.get('temperature', None)
        exp_time = flat_info.get('exposure_time', None)

        with fits.open(file_path) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data.astype(np.float32)

            # Attempt to retrieve the raw TIMESTAMP
            timestamp_raw = hdr.get('TIMESTAMP', hdr.get('HIERARCH TIMESTAMP', 'UNKNOWN'))
            time_str = str(timestamp_raw)

        # Discard saturated
        if data.max() >= saturation_threshold:
            flats_data.append({
                'original_path': file_path,
                'temperature': temp,
                'exposure_time': exp_time,
                'time_header': time_str,
                'data': None,
                'valid': False
            })
            continue

        # Subtract calibration if needed
        if do_calibration:
            #if (bias_map_by_temp is not None) and (temp is not None) and (temp in bias_map_by_temp):
            #    data = data - bias_map_by_temp[temp]
            if (dark_map_by_temp_exp is not None) and (temp is not None) and (exp_time is not None):
                dmap = find_closest_dark(temp, exp_time, dark_map_by_temp_exp, 1.0, 0.1)
                if dmap is not None:
                    data = data - dmap

        # Normalize by max
        max_val = data.max()
        if max_val > 0:
            data /= max_val

        # Save the normalized flat as FITS
        base_name = os.path.basename(file_path)
        out_fits_name = base_name.replace('.fits', '_norm.fits')
        out_fits_path = os.path.join(norm_dir, out_fits_name)

        # We can keep the same header or create a new one
        hdr['COMMENT'] = "Normalized by max in script."
        # Overwrite or new approach:
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdu.writeto(out_fits_path, overwrite=True)

        flats_data.append({
            'original_path': file_path,
            'temperature': temp,
            'exposure_time': exp_time,
            'time_header': time_str,
            'data': data,  # normalized
            'valid': True
        })

    return flats_data


##############################################################################
# HELPER: Mark outliers in an image
##############################################################################
def overlay_outliers(ax, mask, color='red', marker_size=10):
    coords = np.argwhere(mask)
    if len(coords) > 0:
        ys, xs = coords[:, 0], coords[:, 1]
        ax.scatter(xs, ys, s=marker_size, facecolors='none', edgecolors=color, linewidths=0.8)


##############################################################################
# 5A) ANALYSIS BY TEMPERATURE, referencing the frame with SMALLEST timestamp
##############################################################################
def analyze_by_temperature(flats_data, outlier_sigma=5.0, output_folder="outgasing_analysis"):
    base_outdir = os.path.join(output_folder, "analysis_by_temperature")
    os.makedirs(base_outdir, exist_ok=True)

    # Group by approximate temperature
    temp_groups = defaultdict(list)
    for f in flats_data:
        if not f['valid']:
            continue
        if f['temperature'] is not None:
            temp_rounded = round(f['temperature'], 1)
        else:
            temp_rounded = None
        temp_groups[temp_rounded].append(f)

    analysis_results = []
    outlier_pixels_global = defaultdict(set)

    for T_approx, group in temp_groups.items():
        if not group:
            continue

        # Sort group by numeric timestamp ascending
        group.sort(key=lambda x: (parse_timestamp(x['time_header']) or 1e20))

        # Use the smallest timestamp as reference (now the first in sorted group)
        ref_data = group[0]['data']
        ref_time = group[0]['time_header']

        # Create directory for this temperature
        safe_temp_str = str(T_approx)
        temp_dir = os.path.join(base_outdir, f"T_{safe_temp_str}")
        os.makedirs(temp_dir, exist_ok=True)

        for idx, flat in enumerate(group):
            time_str = flat['time_header']
            fdata = flat['data']
            base_name = os.path.basename(flat['original_path'])

            if idx == 0:
                diff_map = np.zeros_like(fdata)
                std_diff = 0.0
                n_outliers = 0
                mask = np.zeros_like(fdata, dtype=bool)
            else:
                diff_map = fdata - ref_data
                std_diff = np.std(diff_map)
                mask = np.abs(diff_map) > (outlier_sigma * std_diff)
                n_outliers = mask.sum()
                if n_outliers > 0:
                    for (r, c) in np.argwhere(mask):
                        outlier_pixels_global[(r, c)].add((T_approx, time_str))

            # Build filename with the time in name
            safe_time = time_str.replace('.', '_').replace(' ', '_').replace('/', '_')
            diff_filename = os.path.join(temp_dir, f"diffmap_{safe_time}.png")

            # color scale ±3σ
            vmin, vmax = -3*std_diff, 3*std_diff
            if std_diff == 0:
                vmin, vmax = diff_map.min(), diff_map.max()

            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(diff_map, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            short_name = os.path.basename(base_name)
            ax.set_title(f"Diff(T={T_approx}, time={time_str})\nRef={ref_time}, File={short_name}\nOutliers={n_outliers}")
            overlay_outliers(ax, mask, marker_size=15)
            plt.colorbar(im, ax=ax, label="Difference")
            fig.savefig(diff_filename, bbox_inches='tight')
            plt.close()

            analysis_results.append({
                'analysis_type': 'by_temperature',
                'approx_temp': T_approx,
                'file_path': flat['original_path'],
                'time_header': time_str,
                'n_outliers': int(n_outliers),
                'std_diff': float(std_diff),
                'mean_flat': float(np.mean(fdata)),
                'std_flat': float(np.std(fdata))
            })

    return analysis_results, outlier_pixels_global


##############################################################################
# 5B) ANALYSIS IGNORING TEMPERATURE, ref = smallest timestamp overall
##############################################################################
def analyze_ignore_temperature(flats_data, outlier_sigma=10.0, output_folder="outgasing_analysis"):
    base_outdir = os.path.join(output_folder, "analysis_by_time_ignore_temp")
    os.makedirs(base_outdir, exist_ok=True)

    valid_flats = [f for f in flats_data if f['valid']]
    if not valid_flats:
        return [], {}

    # Sort by time to find the earliest
    valid_flats.sort(key=lambda x: (parse_timestamp(x['time_header']) or 1e20))

    ref_data = valid_flats[0]['data']
    ref_time = valid_flats[0]['time_header']

    analysis_results = []
    outlier_pixels_global = defaultdict(set)

    for idx, flat in enumerate(tqdm(valid_flats, desc=f"Computing flat", leave = False)):
        time_str = flat['time_header']
        fdata = flat['data']
        base_name = os.path.basename(flat['original_path'])

        if idx == 0:
            diff_map = np.zeros_like(fdata)
            std_diff = 0.0
            n_outliers = 0
            mask = np.zeros_like(fdata, dtype=bool)
        else:
            diff_map = fdata - ref_data
            std_diff = np.std(diff_map)
            mask = np.abs(diff_map) > (outlier_sigma * std_diff)
            n_outliers = mask.sum()
            if n_outliers > 0:
                for (r, c) in tqdm(np.argwhere(mask), leave=False):
                    outlier_pixels_global[(r, c)].add(('IgnoreTemp', time_str))

        safe_time = time_str.replace('.', '_').replace(' ', '_').replace('/', '_')
        diff_filename = os.path.join(base_outdir, f"diffmap_{safe_time}.png")

        vmin, vmax = -3*std_diff, 3*std_diff
        if std_diff == 0:
            vmin, vmax = diff_map.min(), diff_map.max()

        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(diff_map, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        short_name = os.path.basename(base_name)
        ax.set_title(f"Diff(IgnoreTemp) time={time_str}\nRef={ref_time}, File={short_name}\nOutliers={n_outliers}")
        overlay_outliers(ax, mask, marker_size=15)
        plt.colorbar(im, ax=ax, label="Difference")
        fig.savefig(diff_filename, bbox_inches='tight')
        plt.close()

        analysis_results.append({
            'analysis_type': 'ignore_temperature',
            'approx_temp': None,
            'file_path': flat['original_path'],
            'time_header': time_str,
            'n_outliers': int(n_outliers),
            'std_diff': float(std_diff),
            'mean_flat': float(np.mean(fdata)),
            'std_flat': float(np.std(fdata))
        })

    return analysis_results, outlier_pixels_global


##############################################################################
# 6) CLASSIFY OUTLIERS (simple approach)
##############################################################################
def classify_outliers(global_outlier_dicts):
    """
    We keep the naive logic:
      - If an outlier pixel appears at multiple distinct T => "Sensor Damage".
      - If single T (except 'IgnoreTemp') but multiple times => "Likely Sensor".
      - Else => "Temperature".
    """
    merged_outliers = defaultdict(set)
    for outdict in tqdm(global_outlier_dicts, desc="Clasifying outliers"):
        for px, occur_set in tqdm(outdict.items(), leave=False):
            merged_outliers[px].update(occur_set)

    results = []
    for (r, c), occurrences in tqdm(merged_outliers.items(), leave=False):
        temps = set()
        times = set()
        for (T_label, t_str) in occurrences:
            temps.add(T_label)
            times.add(t_str)
        n_temps = len(temps)
        n_times = len(times)

        if n_temps > 1 and "IgnoreTemp" not in temps:
            classification = "Sensor Damage"
        elif (n_temps == 1 and "IgnoreTemp" not in temps) and (n_times > 1):
            classification = "Likely Sensor"
        else:
            classification = "Temperature"

        # Additional (deeper) analysis idea:
        # e.g., track min & max times, track whether the pixel is drifting in intensity, etc.
        # That can help confirm if it's truly sensor damage or just sporadic thermal effect.

        results.append({
            'pixel': (r, c),
            'num_occurrences': len(occurrences),
            'temp_labels': list(temps),
            'times': list(times),
            'classification': classification
        })
    return results


##############################################################################
# 7) SAVE CSV W/ SAFER DELIMITERS
##############################################################################
def save_analysis_to_csv(analysis_results, classification_results,
                         prefix="analysis"):
    """
    Use semicolons or quotes to avoid messing up columns.
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    flats_csv_path = f"{prefix}_flats.csv"
    with open(flats_csv_path, 'w') as f:
        # Write header
        f.write("analysis_type,approx_temp,file_path,time_header,n_outliers,std_diff,mean_flat,std_flat\n")
        for row in analysis_results:
            analysis_type = row.get('analysis_type','')
            approx_temp = str(row.get('approx_temp',''))
            file_path = row.get('file_path','')
            time_header = row.get('time_header','')
            n_outliers = row.get('n_outliers',0)
            std_diff = row.get('std_diff',0)
            mean_flat = row.get('mean_flat',0)
            std_flat = row.get('std_flat',0)

            # Comma-delimited row
            f.write(f"{analysis_type},{approx_temp},\"{file_path}\",{time_header},{n_outliers},{std_diff},{mean_flat},{std_flat}\n")

    outliers_csv_path = f"{prefix}_outliers.csv"
    with open(outliers_csv_path, 'w') as f:
        f.write("pixel,num_occurrences,temp_labels,times,classification\n")
        for row in tqdm(classification_results):
            (r,c) = row['pixel']
            pixel_str = f"({r},{c})"
            num_occurrences = row['num_occurrences']
            # We'll separate temp_labels & times with ';'
            temps_str = ";".join(str(t) for t in tqdm(row['temp_labels'], leave=False))
            times_str = ";".join(str(tt) for tt in tqdm(row['times'], leave=False))
            classification = row['classification']

            f.write(f"\"{pixel_str}\",{num_occurrences},\"{temps_str}\",\"{times_str}\",{classification}\n")


##############################################################################
# 8A) SUMMARY PLOTS: downsample time ticks & show outliers vs time, etc.
##############################################################################
def create_summary_plots(analysis_results, prefix="summary_plots"):
    outdir = prefix
    os.makedirs(outdir, exist_ok=True)

    combined = analysis_results[:]
    # Sort by time_header as float for ordering
    combined.sort(key=lambda x: (parse_timestamp(x['time_header']) or 1e20))

    times = [str(x['time_header']) for x in combined]
    outliers = [x['n_outliers'] for x in combined]

    # Possibly skip ticks if too many
    MAX_TICKS = 20
    step = max(1, len(times)//MAX_TICKS)

    # Plot outliers vs time (categorical bar)
    plt.figure(figsize=(12,5))
    plt.bar(range(len(times)), outliers)
    plt.title("Outliers vs Time (categorical)")

    # Label only some ticks
    ticks_labels = [times[i] if i % step == 0 else "" for i in range(len(times))]
    plt.xticks(range(len(times)), ticks_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "outliers_vs_time.png"))
    plt.close()

    # Plot outliers vs temperature
    with_temp = [r for r in combined if r['approx_temp'] is not None]
    with_temp.sort(key=lambda x: x['approx_temp'])
    temps = [r['approx_temp'] for r in with_temp]
    outs = [r['n_outliers'] for r in with_temp]

    plt.figure(figsize=(6,5))
    plt.scatter(temps, outs)
    plt.title("Outliers vs Temperature")
    plt.xlabel("Approx Temperature")
    plt.ylabel("Number of Outliers")
    plt.savefig(os.path.join(outdir, "outliers_vs_temperature.png"))
    plt.close()

    # Mean & std vs time
    means = [x['mean_flat'] for x in combined]
    stds  = [x['std_flat']  for x in combined]

    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.set_title("Mean & Std of Flat vs Time (categorical)")
    ax1.plot(range(len(times)), means, 'o-', color='blue')
    ax1.set_ylabel("Mean of flat", color='blue')
    ax1.set_xlabel("Time index (categorical)")

    ax2 = ax1.twinx()
    ax2.plot(range(len(times)), stds, 's--', color='red')
    ax2.set_ylabel("Std of flat", color='red')

    # Skip ticks
    plt.xticks(range(len(times)), ticks_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_std_vs_time.png"))
    plt.close()

    # Mean & std vs temperature
    tvals = [r['approx_temp'] for r in with_temp]
    mvals = [r['mean_flat']   for r in with_temp]
    svals = [r['std_flat']    for r in with_temp]

    fig, ax1 = plt.subplots()
    ax1.set_title("Mean & Std of Flat vs Temperature")
    ax1.set_xlabel("Approx Temp")
    ax1.plot(tvals, mvals, 'o-', color='blue')
    ax1.set_ylabel("Mean of Flat", color='blue')
    ax2 = ax1.twinx()
    ax2.plot(tvals, svals, 's--', color='red')
    ax2.set_ylabel("Std of Flat", color='red')
    plt.savefig(os.path.join(outdir, "mean_std_vs_temp.png"))
    plt.close()


##############################################################################
# 8B) SCATTER OF OUTLIERS BY TEMPERATURE (pixel coords)
##############################################################################
def plot_outliers_by_temperature(classification_results, analysis_results,
                                 prefix="sensor_plane_outliers"):
    """
    For each distinct temperature from the analysis,
    gather all outliers that had that temperature label and produce a scatter
    (row vs col). The marker size or color can reflect how many times it appears.
    We'll do a simple approach: size ~ num_occurrences, color by classification.

    classification_results: list of {pixel, classification, num_occurrences, temp_labels, times}
    analysis_results: we might use it to get sensor dimension, or skip.
    """
    outdir = prefix
    os.makedirs(outdir, exist_ok=True)

    # Collect distinct temperatures from analysis
    temps_all = set()
    for r in analysis_results:
        if r['approx_temp'] is not None:
            temps_all.add(r['approx_temp'])
    temps_list = sorted(list(temps_all))

    # A small color map for classification
    color_map = {
        "Sensor Damage": "red",
        "Likely Sensor": "orange",
        "Temperature":  "green"
    }

    # For each T, gather outliers that mention T in their temp_labels
    for T in temps_list:
        # get outliers referencing T
        T_outliers = [(row, row['classification']) for row in classification_results
                      if str(T) in [str(tt) for tt in row['temp_labels']]]

        if not T_outliers:
            continue

        # Prepare scatter data
        xs = []
        ys = []
        sizes = []
        colors = []
        for (info, cls) in T_outliers:
            (r, c) = info['pixel']  # row, col
            x = c
            y = r
            xs.append(x)
            ys.append(y)
            sizes.append(info['num_occurrences']*20)  # scale marker
            color = color_map.get(cls, "blue")
            colors.append(color)

        plt.figure(figsize=(6,5))
        plt.scatter(xs, ys, s=sizes, c=colors, alpha=0.6, edgecolors='k')
        plt.title(f"Outliers for T={T} (row vs col)\nMarker size ~ #occurrences, color by classification")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.gca().invert_yaxis()  # Often row=0 top
        plt.tight_layout()
        outpath = os.path.join(outdir, f"outliers_T{T}.png")
        plt.savefig(outpath)
        plt.close()


##############################################################################
# 8C) SINGLE PLOT: all outliers in sensor plane, color-coded
##############################################################################
def plot_all_outliers_sensor_plane(classification_results, prefix="sensor_plane_outliers"):
    """
    One big scatter with all outliers in the sensor plane.
    x=col, y=row, color-coded by classification, size by #occurrences.
    """
    outdir = prefix
    os.makedirs(outdir, exist_ok=True)

    color_map = {
        "Sensor Damage": "red",
        "Likely Sensor": "orange",
        "Temperature":  "green"
    }

    xs = []
    ys = []
    sizes = []
    colors = []

    for info in classification_results:
        (r, c) = info['pixel']
        x = c
        y = r
        n_occ = info['num_occurrences']
        cls = info['classification']
        col = color_map.get(cls, 'blue')
        xs.append(x)
        ys.append(y)
        sizes.append(n_occ*20)
        colors.append(col)

    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, s=sizes, c=colors, alpha=0.6, edgecolors='k')
    plt.title("All Outliers in Sensor Plane")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.gca().invert_yaxis()
    # Add legend example (we do a manual legend for each classification):
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Sensor Damage',
               markerfacecolor='none', markeredgecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Likely Sensor',
               markerfacecolor='none', markeredgecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Temperature',
               markerfacecolor='none', markeredgecolor='green', markersize=10)
    ]
    plt.legend(handles=legend_elems)
    plt.tight_layout()
    outpath = os.path.join(outdir, "all_outliers_sensor_plane.png")
    plt.savefig(outpath)
    plt.close()


##############################################################################
# MAIN
##############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Extended script with advanced CSV handling, time downsampling, sensor-plane scatter, user-chosen output folder, ref by smallest timestamp, max-based norm."
    )
    parser.add_argument("basepath", help="Base path containing the FITS files.")
    parser.add_argument("--calibrate", action="store_true", help="Subtract bias/dark from flats before normalizing.")
    parser.add_argument("--output_dir", default="outgasing_analysis",
                        help="Directory where all results (CSV, plots, normalized flats) will be saved.")
    args = parser.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    print("1) Loading observations...")
    manager, long_darks, short_darks = load_observations(args.basepath)

    print("2) Checking bias frames for suspicious zeros...")
    log_path = os.path.join(outdir, 'suspicious_bias.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    good_bias, _ = check_bias_frames(short_darks, log_path=log_path)

    print("3) Generating bias maps...")
    bias_map_by_temp = generate_bias_maps(good_bias)

    print("4.1) Checking darks frames for suspicious zeros...")
    log_path = os.path.join(outdir, 'suspicious_darks.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    long_darks, _ = check_bias_frames(long_darks, log_path=log_path)

    print("4.2) Generating dark maps...")
    dark_map_by_temp_exp = generate_dark_maps(long_darks)

    print("5) Loading and normalizing flats (by max). Calibrate =", args.calibrate)
    flats_data = load_flats(
        manager,
        bias_map_by_temp=bias_map_by_temp,
        dark_map_by_temp_exp=dark_map_by_temp_exp,
        do_calibration=args.calibrate,
        saturation_threshold=60000.0,
        output_folder=outdir
    )
    valid_count = sum(1 for f in flats_data if f['valid'])
    invalid_count = len(flats_data) - valid_count
    print(f"   => {valid_count} valid, {invalid_count} discarded (saturated or missing).")

    print("6) Analyzing by temperature with smallest-timestamp reference...")
    analysis_temp, outliers_temp = analyze_by_temperature(flats_data, outlier_sigma=5.0, output_folder=outdir)

    print("   Analyzing ignoring temperature with smallest-timestamp reference...")
    analysis_ignore, outliers_ignore = analyze_ignore_temperature(flats_data, outlier_sigma=5.0, output_folder=outdir)

    combined_analysis = analysis_temp + analysis_ignore

    print("7) Classifying outliers (simple approach)...")
    classification_results = classify_outliers([outliers_temp, outliers_ignore])

    print("8) Saving CSV files...")
    csv_prefix = os.path.join(outdir, "analysis")
    save_analysis_to_csv(combined_analysis, classification_results, prefix=csv_prefix)

    print("Creating summary plots (downsampled time ticks, etc.)...")
    create_summary_plots(combined_analysis, prefix=os.path.join(outdir, "summary_plots"))

    print("Plotting outliers by temperature in sensor plane...")
    plot_outliers_by_temperature(classification_results, combined_analysis,
                                 prefix=os.path.join(outdir, "sensor_plane_outliers"))

    print("Single plot of all outliers in sensor plane, color-coded by classification...")
    plot_all_outliers_sensor_plane(classification_results,
                                   prefix=os.path.join(outdir, "sensor_plane_outliers"))

    print("\nDEEPER OUTLIER ANALYSIS RECOMMENDATION:")
    print(" - The script still uses a naive classification. For deeper insights,")
    print("   consider tracking outlier intensity drift over time, temperature correlations,")
    print("   repeated persistent pixels, etc. You can store additional stats (e.g. mean diff, local pattern).")
    print("Done!")


if __name__ == "__main__":
    main()
