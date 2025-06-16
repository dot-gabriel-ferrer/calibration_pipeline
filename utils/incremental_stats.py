import numpy as np


def incremental_mean_std(
    frame: np.ndarray, mean: np.ndarray | None, m2: np.ndarray | None, count: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update running mean and M2 for an array using Welford's algorithm.

    Parameters
    ----------
    frame : np.ndarray
        New data array.
    mean : np.ndarray | None
        Current mean array or ``None`` if no samples processed yet.
    m2 : np.ndarray | None
        Current sum of squared differences (M2) array or ``None``.
    count : int
        Number of frames processed so far.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Updated mean array, updated M2 array and new count.
    """
    frame = frame.astype(np.float64)
    if mean is None or m2 is None:
        mean = frame.copy()
        m2 = np.zeros_like(frame, dtype=np.float64)
        return mean, m2, 1

    count += 1
    delta = frame - mean
    mean = mean + delta / count
    delta2 = frame - mean
    m2 = m2 + delta * delta2
    return mean, m2, count
