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


def score_config(
    model: UNetDiffusion,
    config: PurificationConfig,
    x0: torch.Tensor,
    eps: torch.Tensor,
    score_method: ScoreMethod,
):
    # Diffuse original data
    t = torch.full(
        (x0.size(0),),
        config.t_start,
        dtype=torch.long,
        device=x0.device,
    )
    xt, _ = diffuse(x0, t, config.schedule, noise=eps)
    # Generate purification steps
    steps = config.step_curve(config.K, config.t_start, config.schedule)
    # Purify
    x0_hat, _ = reverse_diffusion_ddim(xt, steps, model, config.schedule)

    return score_method(x0, x0_hat)
