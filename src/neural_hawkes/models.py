import torch
import torch.nn as nn

class KernelRowNet(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x):
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if x.ndim == 1:
            x = x.unsqueeze(1)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)