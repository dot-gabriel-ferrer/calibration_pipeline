import numpy as np
from utils.incremental_stats import incremental_mean_std


def test_incremental_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    frames = [rng.random((3, 3)) for _ in range(5)]

    mean = None
    m2 = None
    count = 0
    for f in frames:
        mean, m2, count = incremental_mean_std(f, mean, m2, count)

    stack = np.stack(frames, axis=0)
    assert count == len(frames)
    assert np.allclose(mean, np.mean(stack, axis=0))
    assert np.allclose(np.sqrt(m2 / count), np.std(stack, axis=0))

