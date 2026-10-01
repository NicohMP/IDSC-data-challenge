import numpy as np
import torch
from dataclasses import dataclass
from collections.abc import Callable
from pythresh.thresholds.base import BaseThresholder

# Batch filter response (B, N_angles) →  Batch scores (B,)
Aggregator = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class AnomalyScores:
    horizontal: torch.Tensor  # (B,)
    other: torch.Tensor  # (B,)


@dataclass
class AnomalyDecisions:
    horizontal: np.ndarray  # bool, (B,)
    other: np.ndarray  # bool, (B,)


@dataclass
class DecisionConfig:
    aggregator: Aggregator
    threshold: BaseThresholder
