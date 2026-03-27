import numpy as np


def log_time_transform(t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.log10(np.maximum(t, eps))


def inverse_log_time_transform(t_log: np.ndarray) -> np.ndarray:
    t_log = np.asarray(t_log, dtype=float)
    return 10 ** t_log


def zscore_marks(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std, mean, std


def discretize_marks(marks: np.ndarray,n_bins: int,strategy: str = "quantile"):
    marks = np.asarray(marks, dtype=float)
    if n_bins <= 0:
        raise ValueError("Problem with n_bins, i.e. it must be positive")
    if marks.size == 0:
        raise ValueError("Problem with marks, i.e. it must be non-empty.")

    if strategy == "quantile":
        bin_edges = np.quantile(marks, np.linspace(0.0, 1.0, n_bins + 1))
    elif strategy == "uniform":
        bin_edges = np.linspace(marks.min(), marks.max(), n_bins + 1)
    else:
        raise ValueError("Problem with the strategy specification; it must be 'quantile' or 'uniform'.")

    if np.any(np.diff(bin_edges) <= 0):
        raise ValueError("Bin edges are not strictly increasing. ")

    marks_binned = np.digitize(marks, bin_edges[1:-1], right=False)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return marks_binned.astype(int), bin_edges, bin_centers