import numpy as np
from radiation_analysis import diff_heatmap


def test_diff_heatmap_saves_npz(tmp_path):
    ref = np.zeros((2, 2), dtype=float)
    targ = np.ones((2, 2), dtype=float)
    out_png = tmp_path / "diff.png"
    diff_heatmap(ref, targ, str(out_png), "title")
    assert (tmp_path / "diff.npz").is_file()
