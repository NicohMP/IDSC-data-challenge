import torch
from network import UNetDiffusion
from inference import DiffusionSchedule, reverse_diffusion_ddim
from diffusion import DiffusionSchedule, diffuse
from .config import PurificationConfig


class Purifier:
    def __init__(
        self,
        model: UNetDiffusion,
        schedule: DiffusionSchedule,
    ) -> None:
        self.model = model.eval()
        self.schedule = schedule

    @torch.inference_mode()
    def reconstruct(
        self,
        x0: torch.Tensor,
        config: PurificationConfig,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.full(
            (x0.size(0),),
            config.t_start,
            dtype=torch.long,
            device=x0.device,
        )

        xt, _ = diffuse(x0, t, self.schedule, noise=eps)

        steps = config.step_curve(
            config.K,
            config.t_start,
            self.schedule,
        )

        x0_hat, _ = reverse_diffusion_ddim(
            xt,
            steps,
            self.model,
            self.schedule,
        )
        return x0_hat
