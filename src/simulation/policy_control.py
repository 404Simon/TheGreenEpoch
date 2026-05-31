"""Pause/resume policy logic for CO2-aware training runs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicyAction(Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class PolicyControl:
    """Evaluate pause/resume decisions from grid carbon intensity.

    Parameters
    ----------
    theta_pause : float
        Carbon-intensity threshold above which running training should pause.
    theta_resume : float
        Carbon-intensity threshold below which paused training may resume.
    """

    theta_pause: float
    theta_resume: float

    def __post_init__(self) -> None:
        if self.theta_resume > self.theta_pause:
            raise ValueError("theta_resume must be less than or equal to theta_pause")

    def evaluate(self, co2_intensity: float, is_paused: bool) -> PolicyAction:
        """Decide what to do for the current CO2 intensity and state.

        Returns
        -------
        PolicyAction
            ``"pause"`` when running and the intensity exceeds ``theta_pause``;
            ``"resume"`` when paused and the intensity drops below ``theta_resume``;
            otherwise ``"continue"``.
        """

        if is_paused:
            if co2_intensity < self.theta_resume:
                return PolicyAction.RESUME
            return PolicyAction.CONTINUE

        if co2_intensity > self.theta_pause:
            return PolicyAction.PAUSE

        return PolicyAction.CONTINUE