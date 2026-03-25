from dataclasses import dataclass

@dataclass
class Config:
    D: int = 2
    M: int = 5
    T: float = 10.0
    h: float = 1.0

    nlin: int = 50
    nlog: int = 100
    n_collocation: int = 512
    n_quadrature: int = 500

    hidden_dim: int = 64
    n_layers: int = 3

    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    short_time_fraction: float = 0.3

    seed: int = 42