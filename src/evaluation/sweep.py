from network import UNetDiffusion
import torch
import numpy as np
from purification.config import PurificationConfig
from .scoring import ScoreMethod, score_config
from utils import normalize_patch


class EvaluationSweep:
    def __init__(
        self,
        model: UNetDiffusion,
        configs: list[PurificationConfig],
        patches: dict[str, np.ndarray],
        annotations: dict[str, dict[str, bool]],
        score_method: ScoreMethod,
        device: torch.device,
        batch_size: int,
        seeds: list[int] | None = None,
    ) -> None:

        self.model = model
        self.configs = configs
        self.patches = patches
        self.annotations = annotations
        self.score_method = score_method
        self.device = device
        self.batch_size = batch_size
        self.seeds = [0] if seeds is None else seeds

    def run(self) -> list[dict[str, object]]:
        """Évalue chaque (patch, configuration, seed)."""
        results: list[dict[str, object]] = []

        patch_names = sorted(self.patches)

        x0_all = torch.stack(
            [
                normalize_patch(
                    torch.from_numpy(self.patches[name].astype(np.float32).squeeze())
                )
                for name in patch_names
            ]
        )[:, None].to(
            self.device
        )  # (N, 1, 48, 48)

        for seed in self.seeds:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(seed)

            # Un bruit par patch, partagé entre toutes les configurations
            eps_all = torch.randn(
                x0_all.shape,
                generator=rng,
                dtype=x0_all.dtype,
                device=self.device,
            )

            for config in self.configs:
                for start in range(0, len(patch_names), self.batch_size):
                    end = min(start + self.batch_size, len(patch_names))

                    names_batch = patch_names[start:end]
                    x0_batch = x0_all[start:end]
                    eps_batch = eps_all[start:end]

                    scores_batch = score_config(
                        model=self.model,
                        config=config,
                        x0=x0_batch,
                        eps=eps_batch,
                        score_method=self.score_method,
                    )

                    scores_batch = scores_batch.detach().cpu().tolist()

                    for patch_name, score in zip(names_batch, scores_batch):
                        annotation = self.annotations[patch_name]

                        results.append(
                            {
                                "patch_name": patch_name,
                                "seed": seed,
                                "config_name": config.name,
                                "t_start": config.t_start,
                                "K": config.K,
                                "step_curve": config.step_curve_name,
                                "score": score,
                                "horizontal": bool(annotation["horizontal"]),
                                "other": bool(annotation["other"]),
                                "any_line": bool(
                                    annotation["horizontal"] or annotation["other"]
                                ),
                            }
                        )

        return results
