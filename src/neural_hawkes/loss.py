from __future__ import annotations
import torch


def transform_time_inputs(times: torch.Tensor, use_log_time_input: bool = True) -> torch.Tensor:
    times = times.to(dtype=torch.float32)
    if use_log_time_input:
        return torch.log10(times.clamp_min(1e-8))
    return times


def transform_mark_inputs(
    marks: torch.Tensor,
    normalize_marks_for_nn: bool = False,
    mark_mean: float = 0.0,
    mark_std: float = 1.0,
) -> torch.Tensor:
    marks = marks.to(dtype=torch.float32)
    if normalize_marks_for_nn:
        return (marks - float(mark_mean)) / max(float(mark_std), 1e-8)
    return marks


def temporal_weights(residuals: torch.Tensor, eps: float = 5.0) -> torch.Tensor:
    """
    Temporal weights from Eq. (22), we compute independently for each output j.
    residuals shape: [N, D], with rows ordered by increasing time.
    """
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape [N, D].")

    sq = residuals.square()
    total = sq.sum(dim=0, keepdim=True).clamp_min(1e-8)
    prefix = torch.cumsum(sq, dim=0) - sq
    weights = torch.exp(-eps * prefix / total)
    weights[0, :] = 1.0
    return weights


def weighted_mse(residuals: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(residuals.square())
    return torch.mean(weights * residuals.square())

def compute_zeta_weights_for_row(G_hat: torch.Tensor,stats_time_grid: torch.Tensor,row_index: int) -> torch.Tensor:
    """
    Compute the optional Eq. (30) weights zeta_{ij}(x) for one fixed row i.
    """
    if G_hat.ndim != 4:
        raise ValueError("G_hat must have shape [D, D, L, M].")
    if stats_time_grid.ndim != 1:
        raise ValueError("stats_time_grid must have shape [L].")

    D1, D2, L, M = G_hat.shape
    if D1 != D2:
        raise ValueError("First two dimensions of G_hat must both equal D.")
    if stats_time_grid.shape[0] != L:
        raise ValueError("stats_time_grid length must match the time dimension of G_hat.")
    if not (0 <= row_index < D1):
        raise ValueError("row_index out of range.")

    # areas[i, j, m] = \int |G_ij(t, x_m)| dt
    abs_g = G_hat.abs().permute(0, 1, 3, 2)  # [D, D, M, L]
    areas = torch.trapezoid(abs_g, x=stats_time_grid, dim=-1)  # [D, D, M]

    inv_areas = 1.0 / areas.clamp_min(1e-8)  # [D, D, M]
    denom = inv_areas.sum(dim=(0, 1)).clamp_min(1e-8)  # [M]

    # row slice: [D, M] to transpose to [M, D]
    zeta_row = (inv_areas[row_index] / denom.unsqueeze(0)).transpose(0, 1).contiguous()
    return zeta_row

def evaluate_model_on_grid(model: torch.nn.Module, time_inputs: torch.Tensor, mark_inputs: torch.Tensor) -> torch.Tensor:
    """
    we evaluate a row model on the full Cartesian product grid.
    """
    if time_inputs.ndim != 1:
        raise ValueError("time_inputs must have shape [L].")
    if mark_inputs.ndim != 1:
        raise ValueError("mark_inputs must have shape [M].")

    L = time_inputs.shape[0]
    M = mark_inputs.shape[0]

    t_mesh = time_inputs.repeat_interleave(M)
    x_mesh = mark_inputs.repeat(L)

    phi_flat = model(t_mesh, x_mesh)
    if phi_flat.ndim != 2:
        raise ValueError("Model output must have shape [L*M, D].")

    D = phi_flat.shape[1]
    return phi_flat.reshape(L, M, D)


def _linear_interp_1d(values: torch.Tensor, grid: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation of a single time series.
    """
    if values.ndim != 1:
        raise ValueError("values must have shape [L].")
    if grid.ndim != 1:
        raise ValueError("grid must have shape [L].")
    if values.shape[0] != grid.shape[0]:
        raise ValueError("values and grid must have the same length.")

    q = queries.clamp(min=grid[0], max=grid[-1])
    idx_right = torch.searchsorted(grid, q, right=False).clamp(min=1, max=grid.numel() - 1)
    idx_left = idx_right - 1

    x0 = grid[idx_left]
    x1 = grid[idx_right]
    y0 = values[idx_left]
    y1 = values[idx_right]

    alpha = (q - x0) / (x1 - x0).clamp_min(1e-12)
    return y0 + alpha * (y1 - y0)


def _linear_interp_rowwise(values: torch.Tensor, grid: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """
    Row-wise linear interpolation.
    """
    if values.ndim != 2:
        raise ValueError("values must have shape [N, L].")
    if grid.ndim != 1:
        raise ValueError("grid must have shape [L].")
    if values.shape[1] != grid.shape[0]:
        raise ValueError("values.shape[1] must equal len(grid).")
    if values.shape[0] != queries.shape[0]:
        raise ValueError("values.shape[0] must equal len(queries).")

    q = queries.clamp(min=grid[0], max=grid[-1])
    idx_right = torch.searchsorted(grid, q, right=False).clamp(min=1, max=grid.numel() - 1)
    idx_left = idx_right - 1
    row_idx = torch.arange(values.shape[0], device=values.device)

    x0 = grid[idx_left]
    x1 = grid[idx_right]
    y0 = values[row_idx, idx_left]
    y1 = values[row_idx, idx_right]

    alpha = (q - x0) / (x1 - x0).clamp_min(1e-12)
    return y0 + alpha * (y1 - y0)


def _linear_interp_matrix_common_queries(values: torch.Tensor, grid: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """
    Interpolate multiple series on the same query vector.
    """
    if values.ndim != 2:
        raise ValueError("values must have shape [M, L].")
    if grid.ndim != 1:
        raise ValueError("grid must have shape [L].")
    if values.shape[1] != grid.shape[0]:
        raise ValueError("values.shape[1] must equal len(grid).")

    q = queries.clamp(min=grid[0], max=grid[-1])
    idx_right = torch.searchsorted(grid, q, right=False).clamp(min=1, max=grid.numel() - 1)
    idx_left = idx_right - 1

    x0 = grid[idx_left]
    x1 = grid[idx_right]
    y0 = values[:, idx_left].transpose(0, 1)  # [Q, M]
    y1 = values[:, idx_right].transpose(0, 1)  # [Q, M]

    alpha = ((q - x0) / (x1 - x0).clamp_min(1e-12)).unsqueeze(1)
    return y0 + alpha * (y1 - y0)


def interpolate_G_row(G_row: torch.Tensor,stats_time_grid: torch.Tensor,query_times: torch.Tensor,mark_bins: torch.Tensor) -> torch.Tensor:
    """
    Interpolate G_{ij}(t, x) for a fixed row i.
    """
    if G_row.ndim != 3:
        raise ValueError("G_row must have shape [D, L, M].")

    D, _, _ = G_row.shape
    outputs = []
    for j in range(D):
        series_by_point = G_row[j].transpose(0, 1)[mark_bins]  # [N, L]
        outputs.append(_linear_interp_rowwise(series_by_point, stats_time_grid, query_times))
    return torch.stack(outputs, dim=1)


def evaluate_model_at_collocation(model: torch.nn.Module,collocation_times: torch.Tensor,collocation_mark_bins: torch.Tensor,mark_bin_centers: torch.Tensor,use_log_time_input: bool,normalize_marks_for_nn: bool,mark_mean: float,mark_std: float) -> torch.Tensor:
    
    x_raw = mark_bin_centers[collocation_mark_bins]
    t_in = transform_time_inputs(collocation_times, use_log_time_input=use_log_time_input)
    x_in = transform_mark_inputs(
        x_raw,
        normalize_marks_for_nn=normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )
    return model(t_in, x_in)


def evaluate_model_on_quadrature_grid(model: torch.nn.Module,quadrature_times: torch.Tensor,mark_bin_centers: torch.Tensor,use_log_time_input: bool,normalize_marks_for_nn: bool,mark_mean: float,mark_std: float) -> torch.Tensor:
    t_in = transform_time_inputs(quadrature_times, use_log_time_input=use_log_time_input)
    x_in = transform_mark_inputs(
        mark_bin_centers,
        normalize_marks_for_nn=normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )
    return evaluate_model_on_grid(model, t_in, x_in)


def compute_integral_terms(phi_quad: torch.Tensor,collocation_times: torch.Tensor,collocation_mark_bins: torch.Tensor,quadrature_times: torch.Tensor,quadrature_weights: torch.Tensor,stats_time_grid: torch.Tensor,G_hat: torch.Tensor,lambda_hat: torch.Tensor,p_mark: torch.Tensor) -> torch.Tensor:
    """
    Here, we compute the Fredholm integral term at sampled collocation points.

    """
    if phi_quad.ndim != 3:
        raise ValueError("phi_quad must have shape [Q, M, D].")
    if G_hat.ndim != 4:
        raise ValueError("G_hat must have shape [D, D, L, M].")

    Q, M, D = phi_quad.shape
    N = collocation_times.shape[0]

    p_mark_t = p_mark.transpose(0, 1)  # [M, D]
    source = phi_quad * p_mark_t.unsqueeze(0) * quadrature_weights.view(Q, 1, 1)
    source_sum_over_z = source.sum(dim=1)  # [Q, D]

    integrals = phi_quad.new_zeros((N, D))

    for n in range(N):
        x_bin = int(collocation_mark_bins[n].item())
        delta = collocation_times[n] - quadrature_times

        pos_mask = delta > 0
        neg_mask = delta < 0

        for k in range(D):
            pos_source_k = source_sum_over_z[pos_mask, k]
            neg_source_k = source[neg_mask, :, k]  # [Q_neg, M]

            for j in range(D):
                contrib = phi_quad.new_tensor(0.0)

                if pos_mask.any():
                    g_pos = _linear_interp_1d(G_hat[k, j, :, x_bin], stats_time_grid, delta[pos_mask])
                    contrib = contrib + torch.sum(pos_source_k * g_pos)

                if neg_mask.any() and lambda_hat[j] > 0:
                    ratio = lambda_hat[k] / lambda_hat[j]
                    g_neg_matrix = ratio * _linear_interp_matrix_common_queries(
                        G_hat[j, k].transpose(0, 1),
                        stats_time_grid,
                        -delta[neg_mask],
                    )  # [Q_neg, M]
                    contrib = contrib + torch.sum(neg_source_k * g_neg_matrix)

                integrals[n, j] = integrals[n, j] + contrib

    return integrals


def compute_row_residuals_on_points(model: torch.nn.Module,collocation_times: torch.Tensor,collocation_mark_bins: torch.Tensor,quadrature_times: torch.Tensor,quadrature_weights: torch.Tensor,mark_bin_centers: torch.Tensor,stats_time_grid: torch.Tensor,G_row: torch.Tensor,G_hat: torch.Tensor,lambda_hat: torch.Tensor,p_mark: torch.Tensor,use_log_time_input: bool,normalize_marks_for_nn: bool,mark_mean: float,mark_std: float) -> torch.Tensor:
    """
    Those are the residuals of Eq. (19) evaluated at sampled collocation points.
    """
    model_term = evaluate_model_at_collocation(
        model=model,
        collocation_times=collocation_times,
        collocation_mark_bins=collocation_mark_bins,
        mark_bin_centers=mark_bin_centers,
        use_log_time_input=use_log_time_input,
        normalize_marks_for_nn=normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )

    phi_quad = evaluate_model_on_quadrature_grid(
        model=model,
        quadrature_times=quadrature_times,
        mark_bin_centers=mark_bin_centers,
        use_log_time_input=use_log_time_input,
        normalize_marks_for_nn=normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )

    target = interpolate_G_row(
        G_row=G_row,
        stats_time_grid=stats_time_grid,
        query_times=collocation_times,
        mark_bins=collocation_mark_bins,
    )

    integral = compute_integral_terms(
        phi_quad=phi_quad,
        collocation_times=collocation_times,
        collocation_mark_bins=collocation_mark_bins,
        quadrature_times=quadrature_times,
        quadrature_weights=quadrature_weights,
        stats_time_grid=stats_time_grid,
        G_hat=G_hat,
        lambda_hat=lambda_hat,
        p_mark=p_mark,
    )

    return target - model_term - integral


def boundary_condition_loss(model: torch.nn.Module,T: float,mark_bin_centers: torch.Tensor,use_log_time_input: bool,normalize_marks_for_nn: bool,mark_mean: float,mark_std: float) -> torch.Tensor:
    """
    Penalize u(T, x) != 0 over all mark bins.
    """
    times = torch.full_like(mark_bin_centers, fill_value=float(T))
    t_in = transform_time_inputs(times, use_log_time_input=use_log_time_input)
    x_in = transform_mark_inputs(
        mark_bin_centers,
        normalize_marks_for_nn=normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )
    values = model(t_in, x_in)
    return torch.mean(values.square())

def fredholm_row_loss_on_points(
    model: torch.nn.Module,
    collocation_times: torch.Tensor,
    collocation_mark_bins: torch.Tensor,
    quadrature_times: torch.Tensor,
    quadrature_weights: torch.Tensor,
    mark_bin_centers: torch.Tensor,
    stats_time_grid: torch.Tensor,
    G_row: torch.Tensor,
    G_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    p_mark: torch.Tensor,
    T: float,
    use_log_time_input: bool,
    normalize_marks_for_nn: bool,
    mark_mean: float,
    mark_std: float,
    weights: torch.Tensor | None = None,
    zeta_weights_by_mark: torch.Tensor | None = None,
    boundary_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = compute_row_residuals_on_points(
        model=model,
        collocation_times=collocation_times,
        collocation_mark_bins=collocation_mark_bins,
        quadrature_times=quadrature_times,
        quadrature_weights=quadrature_weights,
        mark_bin_centers=mark_bin_centers,
        stats_time_grid=stats_time_grid,
        G_row=G_row,
        G_hat=G_hat,
        lambda_hat=lambda_hat,
        p_mark=p_mark,
        use_log_time_input=use_log_time_input,
        normalize_marks_for_nn=normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )

    point_weights = None

    if zeta_weights_by_mark is not None:
        point_weights = zeta_weights_by_mark[collocation_mark_bins]  # [N, D]

    if weights is not None:
        point_weights = weights if point_weights is None else weights * point_weights

    loss = weighted_mse(residuals, weights=point_weights)

    if boundary_weight > 0.0:
        loss = loss + boundary_weight * boundary_condition_loss(
            model=model,
            T=T,
            mark_bin_centers=mark_bin_centers,
            use_log_time_input=use_log_time_input,
            normalize_marks_for_nn=normalize_marks_for_nn,
            mark_mean=mark_mean,
            mark_std=mark_std,
        )

    return loss, residuals