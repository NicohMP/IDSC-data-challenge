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
    map: AnomalyMapper
    filter: Filter


class DetectionPipeline:
    def __init__(
        self,
        purifier: Purifier,
        mapper: AnomalyMapper,
        filter: Filter,
        horizontal_decision: DecisionHead,
        other_decision: DecisionHead,
    ) -> None:

        self.purifier = purifier
        self.mapper = mapper
        self.filter = filter
        self.horizontal_decision = horizontal_decision
        self.other_decision = other_decision

    def score_batch(
        self,
        x0: torch.Tensor,  # (B, 1, H, W), normalized
        eps: torch.Tensor,  # (B, 1, H, W)
        pipe_config: PipelineConfig,
    ) -> AnomalyScores:

        x0_hat: torch.Tensor = self.purifier.reconstruct(
            x0=x0, config=pipe_config.purification, eps=eps
        )
        maps = self.mapper(x0, x0_hat)
        responses = self.filter(maps)
        other_scores = self.other_decision.score(responses)
        horizontal_scores = self.horizontal_decision.score(responses)
        return AnomalyScores(horizontal=horizontal_scores, other=other_scores)

    def predict_batch(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        pipe_config: PipelineConfig,
    ) -> PipelineOutput:
        x0_hat: torch.Tensor = self.purifier.reconstruct(
            x0=x0, config=pipe_config.purification, eps=eps
        )
        maps = self.mapper(x0, x0_hat)
        responses = self.filter(maps)

        self.other_decision.fit(responses)
        other_preds = self.other_decision.predict(responses)

        self.horizontal_decision.fit(responses)
        horizontal_preds = self.horizontal_decision.predict(responses)

        return PipelineOutput(
            AnomalyScores(
                self.horizontal_decision.score(responses),
                self.other_decision.score(responses),
            ),
            AnomalyDecisions(horizontal_preds, other_preds),
        )
