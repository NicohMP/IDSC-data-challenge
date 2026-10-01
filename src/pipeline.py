from dataclasses import dataclass
import torch
from scoring.data_type import AnomalyDecisions, AnomalyScores
from scoring.decision import DecisionHead
from purification.purifier import Purifier
from purification.config import PurificationConfig
from filters.filters import Filter
from anomaly_mapping.residuals import AnomalyMapper


@dataclass
class PipelineOutput:
    scores: AnomalyScores
    decisions: AnomalyDecisions


@dataclass(frozen=True)
class PipelineConfig:
    purification: PurificationConfig
    mapper: AnomalyMapper
    filter: Filter
    horizontal_decision: DecisionHead
    other_decision: DecisionHead


class DetectionPipeline:
    """Run configurable anomaly-detection stages with a shared purifier."""

    def __init__(self, purifier: Purifier) -> None:
        self.purifier = purifier

    @torch.inference_mode()
    def score_batch(
        self,
        x0: torch.Tensor,  # (B, 1, H, W), normalized
        eps: torch.Tensor,  # (B, 1, H, W)
        pipe_config: PipelineConfig,
    ) -> AnomalyScores:

        x0_hat: torch.Tensor = self.purifier.reconstruct(
            x0=x0, config=pipe_config.purification, eps=eps
        )
        maps = pipe_config.mapper(x0, x0_hat)
        responses = pipe_config.filter(maps)
        other_scores = pipe_config.other_decision.score(responses)
        horizontal_scores = pipe_config.horizontal_decision.score(responses)
        return AnomalyScores(horizontal=horizontal_scores, other=other_scores)

    @torch.inference_mode()
    def predict_batch(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        pipe_config: PipelineConfig,
    ) -> PipelineOutput:
        """Score a batch and apply decision heads fitted on calibration data."""
        scores = self.score_batch(x0=x0, eps=eps, pipe_config=pipe_config)

        horizontal_preds = pipe_config.horizontal_decision.predict_scores(
            scores.horizontal
        )
        other_preds = pipe_config.other_decision.predict_scores(scores.other)

        return PipelineOutput(
            scores=scores,
            decisions=AnomalyDecisions(
                horizontal=horizontal_preds,
                other=other_preds,
            ),
        )
