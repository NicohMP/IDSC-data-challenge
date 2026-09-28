"""
Batched purification of raw patches (Algorithm 1) and residual maps (Eq. 10).
"""

import numpy as np
import torch

from diffusion import diffuse
from inference import make_ddim_timesteps, reverse_diffusion_ddim


@torch.no_grad()
def residual_maps(
    patches: np.ndarray,
    model,
    schedule,
    t_start: int,
    ddim_steps: int,
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Standardize raw patches (N, H, W) one by one (Eq. 9), purify them and
    return the residual maps (N, H, W) of Eq. (10).
    """
    torch.manual_seed(seed)
    timesteps = make_ddim_timesteps(ddim_steps, schedule.timesteps, t_start)
    out = np.empty(patches.shape, dtype=np.float32)

    for i in range(0, len(patches), batch_size):
        p = patches[i : i + batch_size].astype(np.float32)
        x0 = (p - p.mean(axis=(1, 2), keepdims=True)) / (
            p.std(axis=(1, 2), keepdims=True) + 1e-8
        )
        x0_t = torch.from_numpy(x0)[:, None].to(device)
        t = torch.full((len(p),), t_start, device=device, dtype=torch.long)
        xt, _ = diffuse(x0_t, t, schedule)
        x0_hat, _ = reverse_diffusion_ddim(xt, timesteps, model, schedule)
        x0_hat = x0_hat[:, 0].cpu().numpy()
        out[i : i + len(p)] = np.abs(x0 - x0_hat) / (1 + np.abs(x0_hat))
    return out
