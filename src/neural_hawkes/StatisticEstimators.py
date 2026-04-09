"""
First-order, second-order and mark-distribution statistics estimators
"""

import sys
import os
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from data import EventData


# ── First-order statistics ─────────────────────────────────────────────────────

def estimate_Lambda(events: EventData) -> np.ndarray:
    """
    For a stationary Hawkes process, \Lambda^i is the empirical rate of component i:
        \Lambda^i = N^i_T / T
    """
    D = int(events.types.max()) + 1
    Lambda = np.zeros(D)
    for i in range(D):
        Lambda[i] = np.sum(events.types == i) / events.horizon
    return Lambda


def recover_mu(Lambda: np.ndarray, norm_Phi: np.ndarray) -> np.ndarray:
    """
    Recover the baseline intensity from the stationary rate:
        \mu = (I - \|\Phi\|) \cdot \Lambda
    """
    Lambda   = np.asarray(Lambda,   dtype=float)
    norm_Phi = np.asarray(norm_Phi, dtype=float)
    return (np.eye(len(Lambda)) - norm_Phi) @ Lambda


# ── Mark distribution ──────────────────────────────────────────────────────────

def estimate_mark_distribution(
    events: EventData,
    n_bins: int = 10,
    strategy: str = "uniform",
):
    """
    Estimate the empirical mark distribution p^j for each component j.

    Returns None if events carry no marks.
    """
    if events.marks is None:
        return None

    D = int(events.types.max()) + 1
    M = int(n_bins)

    mark_grid  = np.zeros((D, M))
    mark_probs = np.zeros((D, M))
    bin_edges  = np.zeros((D, M + 1))
    mark_mean  = np.zeros(D)
    mark_std   = np.zeros(D)

    for j in range(D):
        xi_j = events.marks[events.types == j]
        if len(xi_j) == 0:
            continue

        mark_mean[j] = xi_j.mean()
        mark_std[j]  = xi_j.std() if xi_j.std() > 0 else 1.0

        if strategy == "quantile":
            edges = np.quantile(xi_j, np.linspace(0, 1, M + 1))
            edges = np.unique(edges)
            if len(edges) < M + 1:
                edges = np.linspace(xi_j.min(), xi_j.max(), M + 1)
        else:
            edges = np.linspace(xi_j.min(), xi_j.max(), M + 1)

        edges[-1] += 1e-10 * abs(edges[-1])

        bin_edges[j, :len(edges)] = edges[: M + 1]
        counts, _ = np.histogram(xi_j, bins=edges)
        probs     = counts / counts.sum()
        centres   = 0.5 * (edges[:-1] + edges[1:])

        mark_grid[j,  : len(centres)] = centres
        mark_probs[j, : len(probs)]   = probs

    return {
        "mark_grid":  mark_grid,
        "mark_probs": mark_probs,
        "bin_edges":  bin_edges,
        "mark_mean":  mark_mean,
        "mark_std":   mark_std,
    }


# ── Time grid ─────────────────────────────────────────────────────────────────

def build_time_grid(T: float, h: float, n_lin: int, n_log: int) -> np.ndarray:
    """
    Hybrid linear/logarithmic lag grid (Bacry et al.).

    Linear on [t_min, h] with t_min = h/n_lin, then logarithmic on [h, T].
    """
    if not (0 < h < T):
        raise ValueError("Need 0 < h < T.")
    if n_lin < 1 or n_log < 1:
        raise ValueError("n_lin and n_log must be >= 1.")

    t_min = h / n_lin
    lin_part = np.array([t_min + k * (h - t_min) / n_lin for k in range(n_lin)])
    log_part = np.array([h * (T / h) ** ((k + 1) / n_log) for k in range(n_log)])
    return np.unique(np.concatenate([lin_part, log_part]))


# ── Second-order statistics ────────────────────────────────────────────────────

def estimate_G(
    events: EventData,
    Lambda: np.ndarray,
    lag_max: float,
    h: float,
    n_lin: int = 10,
    n_log: int = 50,
    mark_bins=None,
    mark_edges=None,
):
    """
    Estimate the second-order statistics G on a hybrid linear/log time grid.

    Parameters
    ----------
    events    : EventData — sorted event stream.
    Lambda    : ndarray shape (D,) — mean intensities.
    lag_max   : float — maximum lag.
    h         : float — crossover for the linear/log grid.
    n_lin, n_log : int — grid resolution.
    mark_bins, mark_edges : optional mark discretization objects.

    Returns
    -------
    G        : ndarray (D,D,n_grid) or (D,D,M,n_grid)
    t_grid   : ndarray (n_grid,)
    interp_G : callable(i, j, t_query, m=None) → array
    """
    Lambda = np.asarray(Lambda, dtype=float)
    D      = int(events.types.max()) + 1
    marked = (events.marks is not None) and (mark_bins is not None)

    t_grid = build_time_grid(lag_max, h, n_lin, n_log)
    n_grid = len(t_grid)
    edges  = np.concatenate(([0.0], t_grid))
    dt     = np.diff(edges)

    if marked:
        M       = mark_bins.shape[1]
        G_raw   = np.zeros((D, D, M, n_grid), dtype=float)
        cnt_ref = np.zeros((D, M), dtype=int)
    else:
        M       = 1
        G_raw   = np.zeros((D, D, n_grid), dtype=float)
        cnt_ref = np.zeros(D, dtype=int)

    eligible      = events.times <= (events.horizon - lag_max)
    times_by_comp = [events.times[events.types == i] for i in range(D)]

    for k in range(events.n_events):
        if not eligible[k]:
            continue

        t_k = events.times[k]
        j_k = int(events.types[k])

        if marked:
            xi_k    = events.marks[k]
            edges_j = mark_edges[j_k]
            m_k     = int(np.clip(np.searchsorted(edges_j, xi_k, side="right") - 1, 0, M - 1))
            cnt_ref[j_k, m_k] += 1
        else:
            cnt_ref[j_k] += 1

        for i in range(D):
            t_i   = times_by_comp[i]
            left  = np.searchsorted(t_i, t_k,            side="right")
            right = np.searchsorted(t_i, t_k + lag_max,  side="right")
            lags  = t_i[left:right] - t_k

            bin_idx = np.searchsorted(edges, lags, side="right") - 1
            valid   = (bin_idx >= 0) & (bin_idx < n_grid)

            if marked:
                for b in bin_idx[valid]:
                    G_raw[i, j_k, m_k, b] += 1.0
            else:
                for b in bin_idx[valid]:
                    G_raw[i, j_k, b] += 1.0

    # Normalize: conditional density minus baseline
    if marked:
        for i in range(D):
            for j in range(D):
                for m in range(M):
                    n_ref = cnt_ref[j, m]
                    if n_ref == 0:
                        continue
                    G_raw[i, j, m, :] = G_raw[i, j, m, :] / (n_ref * dt) - Lambda[i]
    else:
        for i in range(D):
            for j in range(D):
                n_ref = cnt_ref[j]
                if n_ref == 0:
                    continue
                G_raw[i, j, :] = G_raw[i, j, :] / (n_ref * dt) - Lambda[i]

    G = G_raw

    def interp_G(i, j, t_query, m=None):
        t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
        if marked:
            if m is None:
                raise ValueError("Provide m for marked data.")
            y = G[i, j, m, :]
        else:
            y = G[i, j, :]
        f = interp1d(t_grid, y, kind="linear",
                     bounds_error=False, fill_value=(float(y[0]), 0.0))
        return f(t_query)

    return G, t_grid, interp_G


# ── H kernel ──────────────────────────────────────────────────────────────────

def build_H(G: np.ndarray, t_grid: np.ndarray, Lambda: np.ndarray):
    """
    Build the two-sided kernel H^{kj}(t, x, z) (Eq. 10):

        H^{kj}(t, x_m, z_l) = G^{kj}_m(t)          for t > 0
                              (\Lambda^k/\Lambda^j) G^{jk}_l(−t) for t < 0

    Returns a callable H(k, j, t, m=0, l=0).
    """
    Lambda = np.asarray(Lambda, dtype=float)
    marked = G.ndim == 4

    t_full = np.concatenate([-t_grid[::-1], t_grid])

    def _interp_g(i, j, m=0):
        y_pos  = G[i, j, m, :] if marked else G[i, j, :]
        y_full = np.concatenate([y_pos[::-1], y_pos])
        return interp1d(t_full, y_full, kind="linear",
                        bounds_error=False, fill_value=0.0)

    def H_func(k, j, t, m=0, l=0):
        t   = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros_like(t)
        ratio = Lambda[k] / Lambda[j] if Lambda[j] > 0 else 0.0

        pos = t > 0
        neg = t < 0
        if pos.any():
            out[pos] = _interp_g(k, j, m)(t[pos])
        if neg.any():
            out[neg] = ratio * _interp_g(j, k, l)(-t[neg])
        return out.squeeze()

    return H_func


# ── Optional spline smoothing ─────────────────────────────────────────────────

def smooth_G_spline(
    G: np.ndarray,
    t_grid: np.ndarray,
    smooth_factor=None,
) -> np.ndarray:
    """
    Smooth each G^{ij}(t) curve with a spline in log-time.
    Works on unmarked G of shape (D, D, n_t).
    """
    G      = np.asarray(G, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    D      = G.shape[0]
    G_smooth = np.zeros_like(G)
    x = np.log(t_grid)
    for i in range(D):
        for j in range(D):
            spl = UnivariateSpline(x, G[i, j, :], s=smooth_factor)
            G_smooth[i, j, :] = spl(x)
    return G_smooth
