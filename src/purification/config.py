from dataclasses import dataclass
from diffusion import DiffusionSchedule

from .step_curves import StepCurve


@dataclass
class PurificationConfig:
    """Paramètres d'un débruitage DDIM avec un calendrier donné."""

    t_start: int
    K: int
    schedule: DiffusionSchedule
    step_curve: StepCurve

    @property
    def step_curve_name(self) -> str:
        return self.step_curve.__name__.removesuffix("_timesteps")

    @property
    def name(self) -> str:
        return f"t{self.t_start}_k{self.K}_{self.step_curve_name}"
