from collections.abc import Callable

import numpy as np
import torch
from .data_type import DecisionConfig


class DecisionHead:
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

    def fit(self, responses: torch.Tensor) -> "DecisionHead":
        """Fit the PyThresh method on calibration responses."""
        return self.fit_scores(self.score(responses))

    def fit_scores(self, scores: torch.Tensor) -> "DecisionHead":
        """Fit the PyThresh method on precomputed calibration scores."""
        self.thresholder.fit(scores.detach().cpu().numpy())
        return self

    def predict(self, responses: torch.Tensor) -> np.ndarray:
        """Return one binary decision per patch.
        Returns a numpy array on cpu to match against annotations
        """
        return self.predict_scores(self.score(responses))

    def predict_scores(self, scores: torch.Tensor) -> np.ndarray:
        """Threshold precomputed scores into one binary decision per patch."""
        labels = self.thresholder.predict(scores.detach().cpu().numpy())
        return np.asarray(labels, dtype=bool)
