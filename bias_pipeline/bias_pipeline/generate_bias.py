# Author: Elías Gabriel Ferrer Jorge

"""
Utility Script: Generate Synthetic Bias FITS from Model

This script generates a synthetic bias frame using a temperature-dependent pixel-wise model,
composed of intercept (a_map) and slope (b_map) coefficients. It takes a temperature value
as input and outputs the corresponding synthetic bias frame.

Usage:
------
You must provide:
  - --a-map: Path to FITS file containing a_map (bias intercepts)
  - --b-map: Path to FITS file containing b_map (temperature slopes)
  - --temperature: Target temperature (in °C)
  - --output: Output FITS file or directory

If a directory is provided, a default filename will be constructed: "synthetic_bias_XX.XC.fits"
"""

import argparse
import os
from astropy.io import fits
import numpy as np
from steps.step4_generate_synthetic_bias import generate_synthetic_bias

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bias from model and temperature.")
    parser.add_argument("--a-map", type=str, required=True, help="Path to bias_a_map.fits")
    parser.add_argument("--b-map", type=str, required=True, help="Path to bias_b_map.fits")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature (in °C) to generate bias for.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output FITS path or directory where to save the synthetic bias.")

    args = parser.parse_args()

    print(f"\nGenerating synthetic bias at {args.temperature:.1f}°C...")

    # Load model maps (bias = a + b*T)
    a_map = fits.getdata(args.a_map).astype(np.float32)
    b_map = fits.getdata(args.b_map).astype(np.float32)

    # Compute synthetic bias
    synthetic_bias = generate_synthetic_bias(a_map, b_map, args.temperature)

    # Print basic stats
    mean_val = np.nanmean(synthetic_bias)
    std_val = np.nanstd(synthetic_bias)
    print(f"→ Bias stats: mean = {mean_val:.2f} ADU | std = {std_val:.2f} ADU")

    # Handle output filename and create directory
    if os.path.isdir(args.output):
        filename = f"synthetic_bias_{args.temperature:.1f}C.fits"
        output_path = os.path.join(args.output, filename)
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fits.writeto(output_path, synthetic_bias.astype(np.float32), overwrite=True)

    print(f"Synthetic bias saved to: {output_path}\n")

if __name__ == "__main__":
    main()
