import numpy as np
from .data import EventData

def estimate_first_order_stats(events: EventData, D: int):
    counts = np.bincount(events.types, minlength=D)
    horizon = max(events.times.max(), 1e-8)
    return counts / horizon

def estimate_second_order_stats(events: EventData, D: int, time_grid, mark_grid):
    """
    Placeholder:
    return G with shape [D, D, len(time_grid), len(mark_grid)]
    """
    return np.zeros((D, D, len(time_grid), len(mark_grid)))