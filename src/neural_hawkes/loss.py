import torch

def mse_loss(residuals: torch.Tensor) -> torch.Tensor:
    return torch.mean(residuals ** 2)

def temporal_weights(residuals: torch.Tensor, eps: float = 5.0) -> torch.Tensor:
    """
    residuals shape: [N, D]
    """
    sq = residuals ** 2
    cumsum = torch.cumsum(sq, dim=0)
    total = cumsum[-1:].clamp_min(1e-8)
    weights = torch.exp(-eps * cumsum / total)

    first_row = torch.ones((1,weights.shape[1]),dtype=weights.dtype, device = weights.device)
    if weights.shape[0]==1:
        return first_row 
    
    return torch.cat([first_row, weights[1:]], dim=0)

def weighted_loss(residuals: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(residuals ** 2)
    return torch.mean(weights * residuals ** 2)

def flatten_residuals_time_major(residuals_grid: torch.Tensor) -> torch.Tensor:
    """
    Flatten residuals from [L, M, D] to [L*M, D], keeping time-major order.
    """
    if residuals_grid.ndim != 3:
        raise ValueError("residuals_grid must have shape [L, M, D].")

    L, M, D = residuals_grid.shape
    return residuals_grid.reshape(L * M, D)


def evaluate_model_on_grid(model: torch.nn.Module,time_inputs: torch.Tensor,mark_inputs: torch.Tensor) -> torch.Tensor:
    """
    we evaluate a row-model on the full Cartesian product grid.
    """
    if time_inputs.ndim != 1:
        raise ValueError("time_inputs must have shape [L].")
    if mark_inputs.ndim != 1:
        raise ValueError("mark_inputs must have shape [M].")

    L = time_inputs.shape[0]
    M = mark_inputs.shape[0]

    t_mesh = time_inputs.repeat_interleave(M)        # [L*M]
    x_mesh = mark_inputs.repeat(L)                   # [L*M]

    phi_flat = model(t_mesh.unsqueeze(1), x_mesh.unsqueeze(1))  # [L*M, D]
    if phi_flat.ndim != 2:
        raise ValueError("Model output must have shape [B, D].")

    D = phi_flat.shape[1]
    phi_grid = phi_flat.reshape(L, M, D)
    return phi_grid


def compute_discrete_integral_term(phi_grid: torch.Tensor,H_hat: torch.Tensor,p_mark: torch.Tensor,quad_weights: torch.Tensor,t_idx: int,x_bin: int) -> torch.Tensor:
    """
    we compute the discrete convolution term for one fixed grid point (t_idx, x_bin).
    For a fixed row i, this approximates:
        sum_k sum_r sum_z phi_{ik}(s_r, z) H_{kj}(t - s_r, x, z) p_k(z) w_r
    """
    if phi_grid.ndim != 3:
        raise ValueError("phi_grid must have shape [L, M, D].")
    if H_hat.ndim != 5:
        raise ValueError("H_hat must have shape [D, D, U, M, M].")
    if p_mark.ndim != 2:
        raise ValueError("p_mark must have shape [D, M].")
    if quad_weights.ndim != 1:
        raise ValueError("quad_weights must have shape [L].")

    L, M, D = phi_grid.shape
    D1, D2, U, Mx, Mz = H_hat.shape

    if D1 != D or D2 != D:
        raise ValueError("First two dimensions of H_hat must match D from phi_grid.")
    if Mx != M or Mz != M:
        raise ValueError("Mark dimensions of H_hat must match M from phi_grid.")
    if p_mark.shape != (D, M):
        raise ValueError("p_mark must have shape [D, M].")
    if quad_weights.shape[0] != L:
        raise ValueError("quad_weights must have shape [L].")

    zero_idx = (U - 1) // 2
    integral = phi_grid.new_zeros(D)  # vector over j

    for s_idx in range(L):
        u_idx = zero_idx + (t_idx - s_idx)
        if u_idx < 0 or u_idx >= U:
            continue

        w_s = quad_weights[s_idx]

        for k in range(D):
            for z_bin in range(M):
                phi_val = phi_grid[s_idx, z_bin, k]
                weight_kz = p_mark[k, z_bin]

                # H_slice_j = H_hat[k, j, u_idx, x_bin, z_bin] over j
                H_slice_j = H_hat[k, :, u_idx, x_bin, z_bin]  # shape [D]

                integral = integral + w_s * phi_val * weight_kz * H_slice_j

    return integral


def compute_row_residuals(phi_grid: torch.Tensor,G_row: torch.Tensor,H_hat: torch.Tensor,p_mark: torch.Tensor,quad_weights: torch.Tensor) -> torch.Tensor:
    """
    we compute Fredholm residuals for one fixed row i on the full grid.

    Residual:
        R_{ij}(t, x) = G_{ij}(t, x)
                       - phi_{ij}(t, x)
                       - integral_term_j(t, x)

    """
    if phi_grid.ndim != 3:
        raise ValueError("phi_grid must have shape [L, M, D].")
    if G_row.ndim != 3:
        raise ValueError("G_row must have shape [D, L, M].")

    L, M, D = phi_grid.shape

    if G_row.shape != (D, L, M):
        raise ValueError("G_row must have shape [D, L, M].")

    residuals = phi_grid.new_zeros((L, M, D))

    for t_idx in range(L):
        for x_bin in range(M):
            model_term = phi_grid[t_idx, x_bin, :]         # shape [D]
            target = G_row[:, t_idx, x_bin]                # shape [D]
            integral_term = compute_discrete_integral_term(
                phi_grid=phi_grid,
                H_hat=H_hat,
                p_mark=p_mark,
                quad_weights=quad_weights,
                t_idx=t_idx,
                x_bin=x_bin,
            )                                              # shape [D]

            residuals[t_idx, x_bin, :] = target - model_term - integral_term

    return residuals


def fredholm_row_loss(phi_grid: torch.Tensor,G_row: torch.Tensor,H_hat: torch.Tensor,p_mark: torch.Tensor,quad_weights: torch.Tensor,weight_eps: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full row-wise Fredholm loss
    """
    residuals_grid = compute_row_residuals(phi_grid=phi_grid,G_row=G_row,H_hat=H_hat,p_mark=p_mark,quad_weights=quad_weights)
    residuals_flat = flatten_residuals_time_major(residuals_grid)

    if weight_eps is None:
        loss = mse_loss(residuals_flat)
    else:
        weights = temporal_weights(residuals_flat, eps=weight_eps)
        loss = weighted_loss(residuals_flat, weights)

    return loss, residuals_grid