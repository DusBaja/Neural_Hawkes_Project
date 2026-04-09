from __future__ import annotations

import random
from dataclasses import asdict,dataclass
from typing import Any

import numpy as np
import torch
import copy 

from data import load_csv
from loss import (compute_zeta_weights_for_row,evaluate_model_on_grid,fredholm_row_loss_on_points,temporal_weights,transform_mark_inputs,transform_time_inputs)
from models import KernelRowNet
from preprocessing import discretize_marks
from statisticss import (build_time_grid,estimate_first_order_stats,estimate_mark_pmf,estimate_second_order_stats)

@dataclass
class Config:
    D: int = 2
    M: int = 5#

    # (Eq. 28 in the paper)
    T: float = 10.0
    h: float = 1.0
    t_min: float | None = None   # if None, we use h / nlin as in the paper's numerical experiments
    nlin: int = 50#8
    nlog: int = 100# 8

    # Quadrature grid for the Fredholm integral
    n_quadrature: int = 250 #here for test
    quadrature_grid_type: str = "log"  # "log" or "linear"

    # Mark discretization
    mark_bin_strategy: str = "quantile"  # "quantile" or "uniform"
    use_exact_mark_grid: bool = False 
    
    # DGM network hyperparameters
    hidden_dim: int = 64
    dgm_layers: int = 1

    # (Table 1 of the paper)
    n_collocation: int = 1024#64
    n_validation: int = 128 #16
    batch_size: int = 8
    epochs: int = 1000 #5to test /debeug
    learning_rate: float = 1e-3
    weight_eps: float = 5.0
    short_time_fraction: float = 0.3

    # Optional boundary penalty would be u(T, x) = 0
    boundary_weight: float = 0.1

    # Optional Eq. (29)-(30) weighting for kernels with very different magnitudes
    use_cross_kernel_weighting: bool = False

    seed: int = 42
    device: str = "cpu"
    use_log_time_input: bool = True
    normalize_marks_for_nn: bool = True
    print_every: int = 25#1

    #enforce_positive_kernel: bool = False # not in the paper, but should stabilize f_{ij} for our test/synthetic data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def resolve_t_min(config: Config) -> float:
    if config.t_min is None:
        return float(config.h / config.nlin)
    return float(config.t_min)

def build_quadrature_grid(T: float, t_min: float, n_quadrature: int, grid_type: str = "log") -> tuple[np.ndarray, np.ndarray]:
    """
    Build quadrature centers and weights over (t_min, T)
    """
    if n_quadrature <= 0:
        raise ValueError("n_quadrature must be positive.")
    if not (0 < t_min < T):
        raise ValueError("Need 0 < t_min < T for the quadrature grid.")

    if grid_type == "log":
        edges = np.logspace(np.log10(t_min), np.log10(T), n_quadrature + 1)
    elif grid_type == "linear":
        edges = np.linspace(t_min, T, n_quadrature + 1)
    else:
        raise ValueError("quadrature_grid_type must be 'log' or 'linear'.")

    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.diff(edges)
    return centers.astype(float), weights.astype(float)

def prepare_mark_grid(events,config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If use_exact_mark_grid=True, marks are already the paper's discrete support {1,...,M}.
    We map them directly to bins {0,...,M-1} and we don't quantile-bin them.
    """
    if config.use_exact_mark_grid:
        raw_marks = np.asarray(events.marks, dtype=float)
        rounded = np.rint(raw_marks).astype(int)

        if not np.allclose(raw_marks, rounded):
            raise ValueError(
                "use_exact_mark_grid=True but marks are not integer-valued. "
                "For paper simulations, marks must be exactly 1,...,M."
            )
        if np.any(rounded < 1) or np.any(rounded > config.M):
            raise ValueError(
                f"Exact paper marks must lie in {{1, ..., {config.M}}}."
            )

        marks_binned = rounded - 1
        bin_centers = np.arange(1, config.M + 1, dtype=float)
        bin_edges = np.arange(0.5, config.M + 1.5, 1.0)
        return marks_binned.astype(int), bin_edges, bin_centers

    return discretize_marks(events.marks,n_bins=config.M,strategy=config.mark_bin_strategy)

def prepare_statistics(data_path: str, config: Config) -> dict[str, Any]:
    """
    Full preprocessing + statistics pipeline
    """
    events = load_csv(data_path)
    events.validate(config.D)

    #marks_binned, bin_edges, bin_centers = discretize_marks(
    #    events.marks,
    #    n_bins=config.M,
    #    strategy=config.mark_bin_strategy,
    #)
    marks_binned, bin_edges, bin_centers = prepare_mark_grid(events, config)
    events.attach_binned_marks(marks_binned)

    effective_t_min = resolve_t_min(config)

    stats_time_edges, stats_time_centers = build_time_grid(
        T=config.T,
        h=config.h,
        t_min=effective_t_min,
        nlin=config.nlin,
        nlog=config.nlog,
    )
    quadrature_times, quadrature_weights = build_quadrature_grid(
        T=config.T,
        t_min=effective_t_min,
        n_quadrature=config.n_quadrature,
        grid_type=config.quadrature_grid_type,
    )

    lambda_hat = estimate_first_order_stats(events, config.D)
    p_mark = estimate_mark_pmf(events, config.D, config.M)
    G_hat = estimate_second_order_stats(events, config.D, config.M, stats_time_edges, lambda_hat)

    return {
        "events": events,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "effective_t_min": effective_t_min,
        "stats_time_edges": stats_time_edges,
        "stats_time_centers": stats_time_centers,
        "quadrature_times": quadrature_times,
        "quadrature_weights": quadrature_weights,
        "lambda_hat": lambda_hat,
        "p_mark": p_mark,
        "G_hat": G_hat,
    }

def build_mark_normalization(bin_centers: np.ndarray, config: Config) -> tuple[float, float]:
    marks = np.asarray(bin_centers, dtype=float)
    if not config.normalize_marks_for_nn:
        return 0.0, 1.0
    mean = float(marks.mean())
    std = float(marks.std() + 1e-8)
    return mean, std


def to_torch(stats: dict[str, Any], config: Config) -> dict[str, torch.Tensor | float]:
    device = torch.device(config.device)
    mark_mean, mark_std = build_mark_normalization(stats["bin_centers"], config)

    tensors: dict[str, torch.Tensor | float] = {
        "mark_bin_centers": torch.tensor(stats["bin_centers"], dtype=torch.float32, device=device),
        "stats_time_centers": torch.tensor(stats["stats_time_centers"], dtype=torch.float32, device=device),
        "quadrature_times": torch.tensor(stats["quadrature_times"], dtype=torch.float32, device=device),
        "quadrature_weights": torch.tensor(stats["quadrature_weights"], dtype=torch.float32, device=device),
        "lambda_hat": torch.tensor(stats["lambda_hat"], dtype=torch.float32, device=device),
        "p_mark": torch.tensor(stats["p_mark"], dtype=torch.float32, device=device),
        "G_hat": torch.tensor(stats["G_hat"], dtype=torch.float32, device=device),
        "mark_mean": mark_mean,
        "mark_std": mark_std,
    }
    return tensors


def sample_collocation_points(config: Config, n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (t_n, x_n) as in Section 2.3.2 of the paper.

    - A fraction S of times are sampled on (t_min, h)
    - The remaining times are sampled on (h, T)
    - Marks are sampled uniformly on the discretized mark universe
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    effective_t_min = resolve_t_min(config)

    n_short = int(np.floor(config.short_time_fraction * n_points))
    n_long = n_points - n_short

    if config.h <= effective_t_min:
        n_short = 0
        n_long = n_points

    times_short = torch.empty((0,), dtype=torch.float32, device=device)
    times_long = torch.empty((0,), dtype=torch.float32, device=device)

    if n_short > 0:
        times_short = effective_t_min + (config.h - effective_t_min) * torch.rand(n_short, device=device)

    if n_long > 0:
        lower = max(config.h, effective_t_min)
        times_long = lower + (config.T - lower) * torch.rand(n_long, device=device)

    times = torch.cat([times_short, times_long], dim=0)
    times, _ = torch.sort(times)

    mark_bins = torch.randint(low=0, high=config.M, size=(n_points,), device=device)

    perm = torch.argsort(times)
    times = times[perm]
    mark_bins = mark_bins[perm]
    return times, mark_bins

def learning_rate_at_epoch(epoch: int, config: Config) -> float:
    frac = float(epoch + 1) / float(config.epochs)
    return float(config.learning_rate * (100.0 ** (-frac)))


def train_one_row(
    row_index: int,
    stats_torch: dict[str, torch.Tensor | float],
    config: Config,
) -> tuple[KernelRowNet, dict[str, list[float] | float | int]]:
    device = torch.device(config.device)

    mark_bin_centers = stats_torch["mark_bin_centers"]
    stats_time_centers = stats_torch["stats_time_centers"]
    quadrature_times = stats_torch["quadrature_times"]
    quadrature_weights = stats_torch["quadrature_weights"]
    lambda_hat = stats_torch["lambda_hat"]
    p_mark = stats_torch["p_mark"]
    G_hat = stats_torch["G_hat"]
    mark_mean = float(stats_torch["mark_mean"])
    mark_std = float(stats_torch["mark_std"])

    assert isinstance(mark_bin_centers, torch.Tensor)
    assert isinstance(stats_time_centers, torch.Tensor)
    assert isinstance(quadrature_times, torch.Tensor)
    assert isinstance(quadrature_weights, torch.Tensor)
    assert isinstance(lambda_hat, torch.Tensor)
    assert isinstance(p_mark, torch.Tensor)
    assert isinstance(G_hat, torch.Tensor)

    model = KernelRowNet(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        output_dim=config.D,
        n_layers=config.dgm_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    G_row = G_hat[row_index]
    zeta_weights_by_mark = None

    if config.use_cross_kernel_weighting:
        zeta_weights_by_mark = compute_zeta_weights_for_row(
            G_hat=G_hat,
            stats_time_grid=stats_time_centers,
            row_index=row_index,
        )

    history: dict[str, list[float] | float | int] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val = float("inf")
    best_epoch = -1
    best_state_dict = None

    for epoch in range(config.epochs):
        lr = learning_rate_at_epoch(epoch, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_times, train_mark_bins = sample_collocation_points(
            config, config.n_collocation, device
        )
        val_times, val_mark_bins = sample_collocation_points(
            config, config.n_validation, device
        )

        model.eval()
        with torch.no_grad():
            _, train_residuals_for_weights = fredholm_row_loss_on_points(
                model=model,
                collocation_times=train_times,
                collocation_mark_bins=train_mark_bins,
                quadrature_times=quadrature_times,
                quadrature_weights=quadrature_weights,
                mark_bin_centers=mark_bin_centers,
                stats_time_grid=stats_time_centers,
                G_row=G_row,
                G_hat=G_hat,
                lambda_hat=lambda_hat,
                p_mark=p_mark,
                T=config.T,
                use_log_time_input=config.use_log_time_input,
                normalize_marks_for_nn=config.normalize_marks_for_nn,
                mark_mean=mark_mean,
                mark_std=mark_std,
                weights=None,
                boundary_weight=0.0,
            )
            train_weights = temporal_weights(
                train_residuals_for_weights, eps=config.weight_eps
            )

        model.train()
        batch_losses: list[float] = []

        for start in range(0, config.n_collocation, config.batch_size):
            stop = min(start + config.batch_size, config.n_collocation)

            optimizer.zero_grad()

            batch_loss, _ = fredholm_row_loss_on_points(
                model=model,
                collocation_times=train_times[start:stop],
                collocation_mark_bins=train_mark_bins[start:stop],
                quadrature_times=quadrature_times,
                quadrature_weights=quadrature_weights,
                mark_bin_centers=mark_bin_centers,
                stats_time_grid=stats_time_centers,
                G_row=G_row,
                G_hat=G_hat,
                lambda_hat=lambda_hat,
                p_mark=p_mark,
                T=config.T,
                use_log_time_input=config.use_log_time_input,
                normalize_marks_for_nn=config.normalize_marks_for_nn,
                mark_mean=mark_mean,
                mark_std=mark_std,
                weights=train_weights[start:stop],
                boundary_weight=config.boundary_weight,
                zeta_weights_by_mark=zeta_weights_by_mark,
            )
            batch_loss.backward()
            optimizer.step()

            batch_losses.append(float(batch_loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss, _ = fredholm_row_loss_on_points(
                model=model,
                collocation_times=val_times,
                collocation_mark_bins=val_mark_bins,
                quadrature_times=quadrature_times,
                quadrature_weights=quadrature_weights,
                mark_bin_centers=mark_bin_centers,
                stats_time_grid=stats_time_centers,
                G_row=G_row,
                G_hat=G_hat,
                lambda_hat=lambda_hat,
                p_mark=p_mark,
                T=config.T,
                use_log_time_input=config.use_log_time_input,
                normalize_marks_for_nn=config.normalize_marks_for_nn,
                mark_mean=mark_mean,
                mark_std=mark_std,
                weights=None,
                boundary_weight=0.0,
                zeta_weights_by_mark=None,
            )

        train_loss_value = float(np.mean(batch_losses))
        val_loss_value = float(val_loss.detach().cpu().item())

        history["train_loss"].append(train_loss_value)
        history["val_loss"].append(val_loss_value)
        history["lr"].append(lr)

        if val_loss_value < best_val:
            best_val = val_loss_value
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            print(
                f"[row {row_index}] epoch {epoch:04d} "
                f"lr={lr:.3e} "
                f"train={train_loss_value:.6e} "
                f"val={val_loss_value:.6e} "
                f"best_val={best_val:.6e} "
                f"best_epoch={best_epoch}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch

    return model, history

def train_all_rows(data_path: str, config: Config | None = None) -> dict[str, Any]:
    if config is None:
        config = Config()

    set_seed(config.seed)

    stats = prepare_statistics(data_path, config)
    stats_torch = to_torch(stats, config)

    models = []
    histories = []

    for row_index in range(config.D):
        print(f"\nTraining row {row_index}/{config.D - 1}")
        model, history = train_one_row(row_index=row_index, stats_torch=stats_torch, config=config)
        models.append(model)
        histories.append(history)

    return {
        "config": asdict(config),
        "stats": stats,
        "models": models,
        "histories": histories,
    }


def predict_row_on_grid(model: KernelRowNet, time_centers: np.ndarray, bin_centers: np.ndarray, config: Config) -> np.ndarray:
    device = torch.device(config.device)

    mark_mean, mark_std = build_mark_normalization(bin_centers, config)

    time_inputs = torch.tensor(time_centers, dtype=torch.float32, device=device)
    mark_inputs = torch.tensor(bin_centers, dtype=torch.float32, device=device)

    time_inputs = transform_time_inputs(time_inputs, use_log_time_input=config.use_log_time_input)
    mark_inputs = transform_mark_inputs(
        mark_inputs,
        normalize_marks_for_nn=config.normalize_marks_for_nn,
        mark_mean=mark_mean,
        mark_std=mark_std,
    )

    model.eval()
    with torch.no_grad():
        phi_grid = evaluate_model_on_grid(model, time_inputs, mark_inputs)

    return phi_grid.cpu().numpy()

#Testing partially here 
#if __name__ == "__main__":
#    config = Config()
#    results = train_all_rows(data_path="data/events.csv", config=config)