import torch
from collections.abc import Callable
from inference import reverse_diffusion_ddim
from diffusion import diffuse
from network import UNetDiffusion
from purification.config import PurificationConfig

ScoreMethod = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mean_residual_score(x0: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
    e = torch.abs(x0 - x0_hat) / (1 + torch.abs(x0_hat))
    return e.flatten(start_dim=1).mean(dim=1)  # forme (B,)
