"""Central configuration for discretization, training, and numerical integration."""
from dataclasses import dataclass


@dataclass
class Config:
    # Problem dimensions
    D: int = 2
    M: int = 5

    # Time horizon and grids
    T: float = 10.0
    h: float = 1.0
    t_min: float = 1e-3
    nlin: int = 50
    nlog: int = 100
    n_collocation: int = 512
    n_quadrature: int = 500
    # short_time_fraction: float = 0.3

    # Grid/discretization choices
    mark_bin_strategy: str = "quantile"
    quadrature_grid_type: str = "log"
    stats_grid_type: str = "hybrid"

    # Model hyperparameters
    hidden_dim: int = 64
    n_layers: int = 3

    # Training hyperparameters
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_eps: float = 5.0#1e-8
    validation_fraction: float = 0.1

    # Runtime / preprocessing
    seed: int = 42
    device: str = "cpu"
    use_log_time_input: bool = False
    train_on_grid_only: bool = True
    normalize_marks_for_nn: bool = False