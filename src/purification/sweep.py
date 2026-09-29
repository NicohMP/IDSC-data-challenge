from network import UNetDiffusion
import torch
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator


from purification.config import PurificationConfig
from purification.purifier import Purifier
from diffusion import DiffusionSchedule
from utils import normalize_patch


@dataclass
class PurificationBatch:
    patch_names: list[str]
    config: PurificationConfig
    seed: int
    x0: torch.Tensor
    x0_hat: torch.Tensor


class PurificationSweep:
    def __init__(
        self,
        model: UNetDiffusion,
        schedule: DiffusionSchedule,
        configs: list[PurificationConfig],
        patches: dict[str, np.ndarray],
        device: torch.device,
        batch_size: int,
        seeds: list[int] | None = None,
    ) -> None:

        self.model = model
        self.schedule = schedule
        self.configs = configs
        self.patches = patches
        self.device = device
        self.batch_size = batch_size
        self.seeds = [0] if seeds is None else seeds

    def run(self) -> Iterator[PurificationBatch]:
        """Évalue chaque (patch, configuration, seed)."""
        # Init common purifier
        purifier = Purifier(self.model, self.schedule)
        # Sort patches
        patch_names = sorted(self.patches)
        # Concatenate all patches into a single tensor
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

                    x0_hat = purifier.reconstruct(
                        x0=x0_batch, config=config, eps=eps_batch
                    )
                    yield PurificationBatch(
                        patch_names=names_batch,
                        config=config,
                        seed=seed,
                        x0=x0_batch,
                        x0_hat=x0_hat,
                    )
