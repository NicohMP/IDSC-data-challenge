"""
All residual mapping methods taking as input x0 and x0_hat
"""

from collections.abc import Callable
import torch

# x0          : (B, 1, H, W)
# x0_hat      : (B, 1, H, W)
# anomaly_map : (B, 1, H, W)
AnomalyMapper = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def mean_residual_score(x0: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
    e = torch.abs(x0 - x0_hat) / (1 + torch.abs(x0_hat))
    return e.flatten(start_dim=1).mean(dim=1)  # forme (B,)
