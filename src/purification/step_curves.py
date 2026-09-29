import torch
import numpy as np
from typing import List, Literal, Type

from inference import (
    load_diffusion_model,
    ddim_reverse_step,
    reverse_diffusion_ddim,
    make_ddim_timesteps,
)
from diffusion import DiffusionSchedule, diffuse
from network import UNetDiffusion
from collections.abc import Callable

StepCurve = Callable[
    [int, int, DiffusionSchedule], list[int]
]  # arguments : K, t_start, schedule ; résultat : timesteps décroissants


def _check_steps_inputs(K: int, t_start: int, schedule: DiffusionSchedule):
    if not 0 <= t_start < schedule.timesteps:
        raise ValueError("t_start must be in [0, schedule.timesteps - 1]")
    if not 1 <= K <= t_start + 1:
        raise ValueError("K must be in [1, t_start + 1]")


def quadratic_timesteps(
    K: int, schedule: DiffusionSchedule, t_start: int, *, curve: float
) -> List[int]:
    _check_steps_inputs(K, t_start, schedule)
    n = K - 1
    slope = (t_start - curve * n**2) / n
    steps = [round((curve * (x**2)) + slope * x) for x in range(K)]
    return steps[::-1]


def linear_timesteps(K: int, t_start: int, schedule: DiffusionSchedule) -> List[int]:
    _check_steps_inputs(K, t_start, schedule)
    linear_steps = make_ddim_timesteps(
        num_ddim_steps=K,
        num_diffusion_steps=schedule.timesteps,
        t_start=t_start,
    )
    return linear_steps


def uniform_log_snr_steps(
    K: int, t_start: int, schedule: DiffusionSchedule
) -> List[int]:
    """Choose K reverse timesteps evenly spaced in the schedule's log-SNR."""
    _check_steps_inputs(K, t_start, schedule)
    if K == 1:
        return [t_start]

    alpha_bar = (
        schedule.alpha_bar[: t_start + 1].detach().to("cpu", dtype=torch.float64)
    )
    log_snr = torch.log(alpha_bar) - torch.log1p(-alpha_bar)
    targets = torch.linspace(
        log_snr[t_start].item(), log_snr[0].item(), K, dtype=torch.float64
    )
    steps = [t_start]
    for i, target in enumerate(targets[1:-1], start=1):
        nearest = int(torch.argmin(torch.abs(log_snr - target)))
        # Reserve one distinct integer timestep for every remaining step.
        lower = K - 1 - i
        upper = steps[-1] - 1
        steps.append(min(max(nearest, lower), upper))
    steps.append(0)
    return steps
