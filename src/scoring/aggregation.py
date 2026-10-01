"""
Methods for aggregating per angle scores produced by filters into a single score.
"""

import torch
from collections.abc import Callable

# Batch filter response (B, N_angles) →  Batch scores (B,)
Aggregator = Callable[[torch.Tensor], torch.Tensor]


def mean_score(response: torch.Tensor) -> torch.Tensor:
    """Average each angle score curve into one score per batch element."""
    return response.flatten(start_dim=1).mean(dim=1)


def max_score(response: torch.Tensor) -> torch.Tensor:
    """Select the strongest response for each batch element."""
    return response.flatten(start_dim=1).amax(dim=1)


def top_quantile_mean_score(
    response: torch.Tensor,
    fraction: float = 0.05,
) -> torch.Tensor:
    """Average the largest response values of each batch element."""
    flattened = response.flatten(start_dim=1)
    k = max(1, round(fraction * flattened.size(1)))
    return flattened.topk(k, dim=1).values.mean(dim=1)
