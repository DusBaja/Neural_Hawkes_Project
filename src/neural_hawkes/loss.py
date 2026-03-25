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
    weights[0] = 1.0
    return weights

def weighted_loss(residuals: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(residuals ** 2)
    return torch.mean(weights * residuals ** 2)