from collections.abc import Callable

import numpy as np
import torch
from .data_type import DecisionConfig


class BinaryDecisionHead:
    """Aggregate filter responses and threshold the resulting scores."""

    def __init__(
        self,
        config: DecisionConfig,
    ) -> None:
        self.aggregator = config.aggregator
        self.thresholder = config.threshold

    def score(self, responses: torch.Tensor) -> torch.Tensor:
        """Convert filter responses (B, N) into scores (B,)."""
        scores = self.aggregator(responses)

        if scores.shape != (responses.shape[0],):
            raise ValueError("The aggregator must return one score per patch")

        return scores

    def fit(self, responses: torch.Tensor) -> "BinaryDecisionHead":
        """Fit the PyThresh method on calibration responses."""
        scores = self.score(responses)
        self.thresholder.fit(scores.detach().cpu().numpy())
        return self

    def predict(self, responses: torch.Tensor) -> torch.Tensor:
        """Return one binary decision per patch."""
        scores = self.score(responses)

        labels = self.thresholder.predict(scores.detach().cpu().numpy())

        return torch.as_tensor(
            np.asarray(labels, dtype=bool),
            device=responses.device,
        )
