import numpy as np
from .data import EventData

def build_time_grid(T: float, h: float, t_min: float, nlin: int, nlog: int):
    """
    We build here a hybrid lag-time grid, ie linear on [t_min,h] and log on [h,T].
    """
    if not (0 < t_min <=h<= T):
        raise ValueError("We need 0 < t_min <= h <= T for the time grid construction")
    if nlin <= 0 or nlog < 0: 
        raise ValueError("We need nlin > 0 and nlog >=0")
    linear_edges = np.linspace(t_min, h, nlin + 1)

    if h < T:
        log_edges = np.logspace(np.log10(h), np.log10(T), nlog + 1)
        time_edges = np.unique(np.concatenate([linear_edges, log_edges[1:]]))
    else:
        time_edges = linear_edges

    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    return time_edges, time_centers

def estimate_first_order_stats(events: EventData, D: int):
    """
    Here, we estimate lambda_hat[i]=count(type=i)/horizon, i.e. the average rate of events of each type.
    """
    counts = np.bincount(events.types, minlength=D)
    horizon = max(events.horizon, 1e-8)
    return counts / horizon

def estimate_second_order_stats(events: EventData, D: int, M: int,time_edges: np.ndarray, lambda_hat:np.ndarray) -> np.ndarray:
    """
    events : EventData - it should be sorted by time and already have marks_binned attached
    D : int - Number of event types
    M : int -Number of mark bins
    time_edges : np.ndarray - Lag bin edges of shape [L+1]
    """
    if events.marks_binned is None:
        raise ValueError("marks_binned is missing, we need to discretize marks first.")

    times = events.times
    types = events.types
    marks_binned = events.marks_binned

    if len(time_edges) < 2:
        raise ValueError("time_edges must have length at least 2.")

    L = len(time_edges) - 1
    T_max = time_edges[-1]
    counts = np.zeros((D, D, L, M), dtype=float)

    # Number of trigger events per (type, mark_bin)
    trigger_counts = np.zeros((D, M), dtype=float)

    n_events = events.n_events

    for r in range(n_events):
        t_r = times[r]
        j = types[r]
        m = marks_binned[r]

        trigger_counts[j, m] += 1.0

        q = r + 1
        while q < n_events:
            delta_t = times[q] - t_r

            if delta_t >= T_max:
                break

            if delta_t < time_edges[0]:
                q += 1
                continue

            ell = np.searchsorted(time_edges, delta_t, side="right") - 1

            if 0 <= ell < L:
                i = types[q]
                counts[i, j, ell, m] += 1.0

            q += 1

    # Normalize by trigger count and bin width
    bin_widths = np.diff(time_edges)  # shape L
    G_hat = np.zeros_like(counts)

    for j in range(D):
        for m in range(M):
            n_triggers = trigger_counts[j, m]
            if n_triggers == 0:
                continue
            raw = counts[:,j,:,m]/(n_triggers * bin_widths[None, :])
            G_hat[:, j, :, m] = raw - lambda_hat[:, None]  # shape [D, L]

    return G_hat

def estimate_mark_pmf(events:EventData, D: int, M: int):
    """
    After the discretization, we want to estimate the mark distribution for each type, i.e. p_mark[k,m] = P(mark in bin m | type = k)
    """
    if events.marks_binned is None:
        raise ValueError("Marks must be binned to estimate p_mark: marks_binned is missing, so we need to discretize marks first.")
    p_mark = np.zeros((D, M), dtype=float)
    for k in range(D):
        mask =events.types==k
        bins_k = events.marks_binned[mask]
        if bins_k.size ==0:
            continue
        counts = np.bincount(bins_k,minlength = M)
        p_mark[k]= counts/counts.sum()
    return p_mark

def build_H_hat(G_hat: np.ndarray, lambda_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    We build a discrete approximation of H from G and lambda_hat.
    """
    if G_hat.ndim != 4:
        raise ValueError("G_hat must have shape [D, D, L, M].")
    if lambda_hat.ndim != 1:
        raise ValueError("lambda_hat must have shape [D].")

    D, D2, L, M = G_hat.shape
    if D != D2:
        raise ValueError("First two dimensions of G_hat must both equal D.")
    if lambda_hat.shape[0] != D:
        raise ValueError("lambda_hat length must match G_hat dimensions.")

    n_u = 2 * L - 1
    lag_offsets = np.arange(-(L - 1), L)   # [-(L-1), ..., 0, ..., L-1]
    zero_idx = L - 1

    H_hat = np.zeros((D, D, n_u, M, M), dtype=float)

    # Positive lags: H[k, j](u, x, z) = G[k, j](u, x), independent of z
    for k in range(D):
        for j in range(D):
            for ell in range(L):
                u_idx = zero_idx + ell +1   # ell=0 starts at "smallest positive lag bin"
                if u_idx >=n_u:
                    break
                gx = G_hat[k, j, ell]    # shape [M] over x_bin
                H_hat[k, j, u_idx, :, :] = gx[:, None]  # broadcast over z_bin

    # Negative lags: H[k, j](u, x, z) = (lambda_k/lambda_j) G[j, k](-u, z), independent of x
    for k in range(D):
        for j in range(D):
            if lambda_hat[j] <= 0:
                continue
            ratio = lambda_hat[k] / lambda_hat[j]

            for ell in range(L):
                u_idx = zero_idx -1- ell
                if u_idx < 0:
                    break
                gz = ratio * G_hat[j, k, ell]   # shape [M] over z_bin
                H_hat[k, j, u_idx, :, :] = gz[None, :]  # broadcast over x_bin

    # Exact zero lag left at zero
    H_hat[:, :, zero_idx, :, :] = 0.0

    return H_hat, lag_offsets