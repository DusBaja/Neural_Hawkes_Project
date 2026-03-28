import torch
import torch.nn as nn


class DGMLayer(nn.Module):
    """
    DGM cell with sigmoid gates and a ReLU.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.z_x = nn.Linear(input_dim, hidden_dim)
        self.z_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.g_x = nn.Linear(input_dim, hidden_dim)
        self.g_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.r_x = nn.Linear(input_dim, hidden_dim)
        self.r_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.h_x = nn.Linear(input_dim, hidden_dim)
        self.h_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.z_x(x) + self.z_h(h))
        g = torch.sigmoid(self.g_x(x) + self.g_h(h))
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        h_tilde = torch.relu(self.h_x(x) + self.h_h(r * h))
        return (1.0 - g) * h_tilde + z * h


class KernelRowNet(nn.Module):
    """
    One DGM network per kernel row i: outputting D coordinates phi_{i.}(t, x)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2, n_layers: int = 1):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [DGMLayer(input_dim=input_dim, hidden_dim=hidden_dim) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if x.ndim == 1:
            x = x.unsqueeze(1)

        inputs = torch.cat([t, x], dim=1)
        h = torch.relu(self.input_layer(inputs))
        for layer in self.layers:
            h = layer(inputs, h)
        return self.output_layer(h)