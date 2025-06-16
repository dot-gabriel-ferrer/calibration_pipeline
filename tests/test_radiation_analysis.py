import numpy as np
import pandas as pd
from radiation_analysis import diff_heatmap, plot_mean_std_vs_time


def test_diff_heatmap_saves_npz(tmp_path):
    ref = np.zeros((2, 2), dtype=float)
    targ = np.ones((2, 2), dtype=float)
    out_png = tmp_path / "diff.png"
    diff_heatmap(ref, targ, str(out_png), "title")
    assert (tmp_path / "diff.npz").is_file()


def test_plot_mean_std_vs_time_creates_png(tmp_path):
    df = pd.DataFrame(
        {
            "TEMP": [10.0, 10.0, 20.0, 20.0],
            "FRAME": [0, 1, 0, 1],
            "MEAN": [1.0, 1.1, 2.0, 2.2],
            "STD": [0.1, 0.1, 0.2, 0.2],
        }
    )
    csv = tmp_path / "stats.csv"
    df.to_csv(csv, index=False)
    out_png = tmp_path / "plot.png"
    plot_mean_std_vs_time(str(csv), str(out_png))
    assert out_png.is_file()
