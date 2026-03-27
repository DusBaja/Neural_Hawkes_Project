from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .config import Config
from .data import load_csv
from .loss import evaluate_model_on_grid, fredholm_row_loss
from .models import KernelRowNet
from .preprocessing import discretize_marks, log_time_transform, zscore_marks
from .statistics import (
    build_H_hat,
    build_time_grid,
    estimate_first_order_stats,
    estimate_mark_pmf,
    estimate_second_order_stats,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_statistics(data_path: str, config: Config) -> dict[str, Any]:
    """
    Full preprocessing + statistics pipeline.

    Returns a dictionary containing:
    - events
    - bin_edges
    - bin_centers
    - time_edges
    - time_centers
    - quad_weights
    - lambda_hat
    - p_mark
    - G_hat
    - H_hat
    - lag_offsets
    """
    events = load_csv(data_path)
    events.validate(config.D)

    marks_binned, bin_edges, bin_centers = discretize_marks(
        events.marks,
        n_bins=config.M,
        strategy=config.mark_bin_strategy,
    )
    events.attach_binned_marks(marks_binned)

    time_edges, time_centers = build_time_grid(
        T=config.T,
        h=config.h,
        t_min=config.t_min,
        nlin=config.nlin,
        nlog=config.nlog,
    )

    # For the first implementation, use bin widths as quadrature weights
    quad_weights = np.diff(time_edges)

    lambda_hat = estimate_first_order_stats(events, config.D)
    p_mark = estimate_mark_pmf(events, config.D, config.M)
    G_hat = estimate_second_order_stats(events, config.D, config.M, time_edges)
    H_hat, lag_offsets = build_H_hat(G_hat, lambda_hat)

    return {
        "events": events,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "time_edges": time_edges,
        "time_centers": time_centers,
        "quad_weights": quad_weights,
        "lambda_hat": lambda_hat,
        "p_mark": p_mark,
        "G_hat": G_hat,
        "H_hat": H_hat,
        "lag_offsets": lag_offsets,
    }


def build_model_inputs(
    time_centers: np.ndarray,
    bin_centers: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build neural-network inputs for the grid.

    Returns
    -------
    time_inputs : np.ndarray, shape [L]
    mark_inputs : np.ndarray, shape [M]
    """
    time_inputs = np.asarray(time_centers, dtype=float)
    mark_inputs = np.asarray(bin_centers, dtype=float)

    if config.use_log_time_input:
        time_inputs = log_time_transform(time_inputs)

    if config.normalize_marks_for_nn:
        mark_inputs, _, _ = zscore_marks(mark_inputs)

    return time_inputs, mark_inputs


def to_torch(
    stats: dict[str, Any],
    config: Config,
) -> dict[str, torch.Tensor]:
    """
    Convert numpy statistics to torch tensors on the configured device.
    """
    device = torch.device(config.device)

    time_inputs_np, mark_inputs_np = build_model_inputs(
        stats["time_centers"],
        stats["bin_centers"],
        config,
    )

    tensors = {
        "time_inputs": torch.tensor(time_inputs_np, dtype=torch.float32, device=device),
        "mark_inputs": torch.tensor(mark_inputs_np, dtype=torch.float32, device=device),
        "quad_weights": torch.tensor(stats["quad_weights"], dtype=torch.float32, device=device),
        "p_mark": torch.tensor(stats["p_mark"], dtype=torch.float32, device=device),
        "G_hat": torch.tensor(stats["G_hat"], dtype=torch.float32, device=device),
        "H_hat": torch.tensor(stats["H_hat"], dtype=torch.float32, device=device),
    }
    return tensors


def train_one_row(
    row_index: int,
    stats_torch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[KernelRowNet, dict[str, list[float]]]:
    """
    Train one neural network for a fixed kernel row i.

    Parameters
    ----------
    row_index : int
        Fixed output row i
    stats_torch : dict
        Torch versions of G_hat, H_hat, p_mark, time_inputs, mark_inputs, quad_weights
    config : Config

    Returns
    -------
    model : KernelRowNet
    history : dict
        Contains loss trajectory.
    """
    device = torch.device(config.device)

    time_inputs = stats_torch["time_inputs"]    # [L]
    mark_inputs = stats_torch["mark_inputs"]    # [M]
    quad_weights = stats_torch["quad_weights"]  # [L]
    p_mark = stats_torch["p_mark"]              # [D, M]
    G_hat = stats_torch["G_hat"]                # [D, D, L, M]
    H_hat = stats_torch["H_hat"]                # [D, D, U, M, M]

    D = config.D

    model = KernelRowNet(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        output_dim=D,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # One fixed row target: shape [D, L, M]
    G_row = G_hat[row_index]

    history = {"loss": []}

    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()

        phi_grid = evaluate_model_on_grid(
            model=model,
            time_inputs=time_inputs,
            mark_inputs=mark_inputs,
        )  # [L, M, D]

        loss, residuals_grid = fredholm_row_loss(
            phi_grid=phi_grid,
            G_row=G_row,
            H_hat=H_hat,
            p_mark=p_mark,
            quad_weights=quad_weights,
            weight_eps=config.weight_eps,
        )

        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.detach().cpu().item()))

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(
                f"[row {row_index}] epoch {epoch:04d} "
                f"loss={history['loss'][-1]:.6e}"
            )

    return model, history


def train_all_rows(
    data_path: str,
    config: Config | None = None,
) -> dict[str, Any]:
    """
    End-to-end training for all rows.

    Returns
    -------
    results : dict
        {
            "config": ...,
            "stats": ...,
            "models": list[KernelRowNet],
            "histories": list[dict],
        }
    """
    if config is None:
        config = Config()

    set_seed(config.seed)

    stats = prepare_statistics(data_path, config)
    stats_torch = to_torch(stats, config)

    models = []
    histories = []

    for row_index in range(config.D):
        print(f"\nTraining row {row_index}/{config.D - 1}")
        model, history = train_one_row(
            row_index=row_index,
            stats_torch=stats_torch,
            config=config,
        )
        models.append(model)
        histories.append(history)

    return {
        "config": asdict(config),
        "stats": stats,
        "models": models,
        "histories": histories,
    }


def predict_row_on_grid(
    model: KernelRowNet,
    time_centers: np.ndarray,
    bin_centers: np.ndarray,
    config: Config,
) -> np.ndarray:
    """
    Evaluate a trained row model on the full grid and return numpy output.

    Returns
    -------
    phi_grid : np.ndarray
        Shape [L, M, D]
    """
    device = torch.device(config.device)

    time_inputs_np, mark_inputs_np = build_model_inputs(
        time_centers=time_centers,
        bin_centers=bin_centers,
        config=config,
    )

    time_inputs = torch.tensor(time_inputs_np, dtype=torch.float32, device=device)
    mark_inputs = torch.tensor(mark_inputs_np, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        phi_grid = evaluate_model_on_grid(model, time_inputs, mark_inputs)

    return phi_grid.cpu().numpy()


if __name__ == "__main__":
    config = Config()
    results = train_all_rows(data_path="data/events.csv", config=config)