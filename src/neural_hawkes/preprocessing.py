import numpy as np

def log_time_transform(t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log10(np.maximum(t, eps))

def zscore_marks(x: np.ndarray):
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std, mean, std