"""Central configuration for the moment-based neural Hawkes estimator."""
from dataclasses import dataclass


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