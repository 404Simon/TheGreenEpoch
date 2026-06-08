"""Training simulation results and state enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class SimState(Enum):
    """Training run states."""

    RUNNING = auto()
    PAUSED = auto()


@dataclass
class SimulationResult:
    """Aggregated output for a single (scenario, region, years,
    start, threshold) combination.

    All durations are in hours, energies in kWh, and emissions in
    kg CO₂eq.
    """

    scenario_description: str
    model: str
    region: str
    historical_years: list[int]
    start_time: datetime
    threshold: float
    hysteresis_margin: float

    # -- timing ----------------------------------------------------------
    total_wall_time_h: float = 0.0
    training_time_h: float = 0.0
    paused_time_h: float = 0.0
    checkpoint_overhead_h: float = 0.0

    # -- energy & emissions ----------------------------------------------
    total_energy_kwh: float = 0.0
    training_energy_kwh: float = 0.0
    paused_energy_kwh: float = 0.0
    checkpoint_energy_kwh: float = 0.0
    total_emissions_kgco2: float = 0.0
    baseline_emissions_kgco2: float = 0.0

    # -- progress --------------------------------------------------------
    tokens_processed: int = 0
    tokens_total: int = 0
    completed: bool = False

    # -- pauses ----------------------------------------------------------
    num_pauses: int = 0

    # -- budget ----------------------------------------------------------
    overhead_budget_pct: float = 0.0
    actual_overhead_pct: float = 0.0
    within_overhead_budget: bool = True

    # -- time series (optional, for plotting) ----------------------------
    timestamps: list[datetime] = field(default_factory=list)
    carbon_intensity_series: list[float] = field(default_factory=list)
    state_series: list[str] = field(default_factory=list)

    # -- diagnostics ------------------------------------------------------
    issues: list[str] = field(default_factory=list)
    stop_reason: str = ""

    # ------------------------------------------------------------------
    @property
    def completion_pct(self) -> float:
        if self.tokens_total == 0:
            return 0.0
        return 100.0 * self.tokens_processed / self.tokens_total

    @property
    def idle_time_h(self) -> float:
        """Total non-training time (paused + checkpoint)."""
        return self.paused_time_h + self.checkpoint_overhead_h

    @property
    def idle_energy_kwh(self) -> float:
        """Total non-training energy (paused + checkpoint)."""
        return self.paused_energy_kwh + self.checkpoint_energy_kwh
    
    @property
    def idle_emissions_kgco2(self) -> float:
        """Total non-training emissions (paused + checkpoint)."""
        return self.paused_energy_kwh * self._avg_carbon_intensity() + self.checkpoint_energy_kwh * self._avg_carbon_intensity()
    
    @property
    def training_efficiency_pct(self) -> float:
        """Percentage of total wall time spent training."""
        if self.total_wall_time_h == 0:
            return 0.0
        return 100.0 * self.training_time_h / self.total_wall_time_h
    
    @property
    def carbon_efficiency_pct(self) -> float:
        """Percentage of total emissions that were from training."""
        if self.total_emissions_kgco2 == 0:
            return 0.0
        return 100.0 * (self.total_emissions_kgco2 - self.idle_emissions_kgco2) / self.total_emissions_kgco2
    
    @property
    def co2_savings_pct(self) -> float:
        """Percentage of emissions saved."""
        if self.baseline_emissions_kgco2 == 0:
            return 0.0
        return 100.0 * (self.baseline_emissions_kgco2 - self.total_emissions_kgco2) / self.baseline_emissions_kgco2

    @property
    def time_overhead_pct(self) -> float:
        """Percentage of time overhead."""
        if self.training_time_h == 0:
            return 0.0
        return 100.0 * (self.total_wall_time_h - self.training_time_h) / self.training_time_h

    # WRONG
    @property
    def score(self) -> float:
        """Custom score combining completion, overhead, and emissions."""
        if not self.completed:
            return 0.0
        if not self.within_overhead_budget:
            return 0.0
        epsilon = 1e-6
        return self.co2_savings_pct / max(self.time_overhead_pct, epsilon)

    @property
    def ok(self) -> bool:
        return self.completed and self.within_overhead_budget and not self.issues

    def __repr__(self) -> str:
        """One-line human-readable summary."""
        years = (
            f"y{self.historical_years[0]}"
            if len(self.historical_years) == 1
            else f"y[{self.historical_years[0]}..{self.historical_years[-1]}]"
        )
        flags = ""
        if not self.completed:
            flags += "✗"
        elif not self.within_overhead_budget:
            flags += "✗BUDGET"
        else:
            flags += "✓"
        if self.issues:
            flags += f" ({len(self.issues)} issue{'s' if len(self.issues) > 1 else ''})"
        return (
            f"[{self.region} {years}] "
            f"θ_pause={self.threshold:.0f} θ_resume={self.hysteresis_margin:.0f} | "
            f"wall={self.total_wall_time_h:.1f}h "
            f"train={self.training_time_h:.1f}h "
            f"pause={self.paused_time_h:.1f}h "
            f"overhead={self.actual_overhead_pct:.1f}% "
            f"pauses={self.num_pauses} "
            f"emissions={self.total_emissions_kgco2:.0f}kg "
            f"{flags}"
        )
