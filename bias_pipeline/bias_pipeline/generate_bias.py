# Author: Elías Gabriel Ferrer Jorge

"""
Utility script to generate a synthetic bias frame from a fitted model and a given temperature.

Supports both file and directory paths for --output:
  - If a directory is given, a filename like "synthetic_bias_XXX.XC.fits" will be used.
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

    # Load model maps
    a_map = fits.getdata(args.a_map).astype(np.float32)
    b_map = fits.getdata(args.b_map).astype(np.float32)

    # Generate bias
    synthetic_bias = generate_synthetic_bias(a_map, b_map, args.temperature)

    # Print stats
    mean_val = np.nanmean(synthetic_bias)
    std_val = np.nanstd(synthetic_bias)
    print(f"→ Bias stats: mean = {mean_val:.2f} ADU | std = {std_val:.2f} ADU")

    # Handle output path
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
