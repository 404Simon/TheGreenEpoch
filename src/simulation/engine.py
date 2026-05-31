"""CO2-aware LLM training simulation engine.

The core state machine steps through grid carbon-intensity data,
evaluates a hysteresis-based pause/resume policy at each step, and
tracks energy consumption, emissions, and training progress.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .models import (
    ScenarioParameters,
    SimulationConfig,
    TrainingRunProfile,
    load_training_profiles,
)
from .physics import emissions_g, energy_wh, tokens_per_second
from .policy_control import PolicyAction, PolicyControl
from .results import SimState, SimulationResult

if TYPE_CHECKING:
    from .grid_data import GridDataProvider

logger = logging.getLogger("simulation.engine")


class SimulationRunner:
    """Orchestrates one or more CO2-aware training simulations.

    Parameters
    ----------
    profiles : dict[str, TrainingRunProfile]
        Model training profiles keyed by name (e.g. ``"Deepseek"``).
    provider : GridDataProvider
        Grid CO2 intensity data source.

    Examples
    --------
    >>> runner = SimulationRunner(profiles, provider)
    >>> results = runner.run_scenarios(scenarios)
    >>> for r in results:
    ...     print(r)
    """

    def __init__(
        self,
        profiles: dict[str, TrainingRunProfile],
        provider: GridDataProvider,
    ) -> None:
        self._profiles = profiles
        self._provider = provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        provider: GridDataProvider | None = None,
    ) -> SimulationRunner:
        """Convenience constructor that loads profiles from *data_dir*."""
        data_dir = Path(data_dir)
        profiles = load_training_profiles(data_dir)
        if provider is None:
            from .grid_data import GridDataProvider

            provider = GridDataProvider(data_dir)
        return cls(profiles, provider)

    def run_scenarios(
        self,
        scenarios: list[ScenarioParameters],
        *,
        record_time_series: bool = False,
    ) -> list[SimulationResult]:
        """Run all scenario / start / threshold combos.

        Each scenario expands into
        ``len(thresholds) × len(start_times)`` individual simulations.
        Historical years are averaged by the grid data provider.
        """
        results: list[SimulationResult] = []
        total = sum(len(s.thresholds) * len(s.start_times) for s in scenarios)
        logger.info(
            "Running %d simulation(s) across %d scenario(s)", total, len(scenarios)
        )

        for scenario in scenarios:
            profile = self._profiles.get(scenario.model)
            if profile is None:
                logger.warning(
                    "Unknown model '%s' in scenario '%s' — skipping",
                    scenario.model,
                    scenario.description,
                )
                continue

            for config in scenario.expand():
                result = self.run_one(
                    profile, config, record_time_series=record_time_series
                )
                results.append(result)
                logger.info(result)

        return results

    def run_one(
        self,
        profile: TrainingRunProfile,
        config: SimulationConfig,
        *,
        record_time_series: bool = False,
    ) -> SimulationResult:
        """Execute a single training simulation.

        Returns a fully populated ``SimulationResult``.
        """
        start_time = config.start_time

        timestamps, carbon = self._provider.timeline(
            config.region, config.historical_years
        )
        if len(timestamps) < 2:
            logger.warning(
                "Insufficient grid data for %s / %s — returning empty result",
                config.region,
                config.historical_years,
            )
            # TODO: raise error here, this should never succeed
            return SimulationResult(
                scenario_description=config.scenario_description,
                model=profile.name,
                region=config.region,
                historical_years=config.historical_years,
                start_time=start_time,
                threshold=config.theta_pause,
                hysteresis_margin=config.theta_resume,
                tokens_total=profile.dataset_tokens,
            )

        # -- derived constants -------------------------------------------
        tps = tokens_per_second(profile.gpu_count)
        train_power_w = profile.gpu_count * profile.gpu_power_train * profile.pue
        pause_power_w = profile.gpu_count * profile.gpu_power_pause * profile.pue

        mean_co2 = float(
            self._provider.year_average(config.region, config.historical_years)
            or carbon.mean()
        )

        # step duration from data granularity (uniform for looping)
        granularity = self._provider.granularity(config.region, config.historical_years)
        if granularity is None:
            if len(timestamps) >= 2:
                diff_ns = int(
                    timestamps[1].astype("int64") - timestamps[0].astype("int64")
                )
                granularity = timedelta(microseconds=diff_ns / 1000)
            else:
                granularity = timedelta(minutes=5)
        step_s = granularity.total_seconds()

        # -- locate starting index ---------------------------------------
        start_idx = _find_start_index(timestamps, start_time)
        idx = start_idx
        n_points = len(timestamps)

        # -- state machine -----------------------------------------------
        policy = PolicyControl(
            theta_pause=config.theta_pause, theta_resume=config.theta_resume
        )

        # TODO: shouldnt we start pausing?? lets discuss this? what if the threshold isnt met initially?
        state = SimState.RUNNING
        transition_timer_s: float = 0.0
        target_after_transition: SimState | None = None

        tokens_remaining = profile.dataset_tokens

        # accumulators (seconds, Wh, g CO2)
        total_wall_s: float = 0.0
        training_s: float = 0.0
        paused_s: float = 0.0
        checkpoint_s: float = 0.0
        total_energy_wh: float = 0.0
        training_energy_wh: float = 0.0
        paused_energy_wh: float = 0.0
        checkpoint_energy_wh: float = 0.0
        total_emissions_g: float = 0.0
        num_pauses: int = 0

        # termination guards
        max_iterations = 10_000_000
        iterations = 0

        # synthetic wall clock (advances with simulation time)
        wall_clock = start_time

        # time series
        ts_timestamps: list[datetime] = []
        ts_carbon: list[float] = []
        ts_state: list[str] = []

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        while tokens_remaining > 0:
            iterations += 1
            if iterations > max_iterations:
                # lets add a field to the SimulationResult why the run was stopped (successfully, myx_iteration_limit, etc.)
                logger.warning(
                    "Iteration limit reached for %s / %s — aborting",
                    config.region,
                    config.historical_years,
                )
                break

            # budget check: stop if overhead already exceeded allowance
            if training_s > 0:
                # Same here, we need to handle this to be able to read off why the hell the training wasnt fully carried out: lets add a field to the SimulationResult why the run was stopped (successfully, myx_iteration_limit, etc.)
                overhead_s = paused_s + checkpoint_s
                overhead_pct = 100.0 * overhead_s / training_s
                if overhead_pct > config.overhead_budget_pct:
                    logger.info(
                        "Overhead %.1f%% exceeds budget %.0f%% — "
                        "stopping simulation for %s / %s",
                        overhead_pct,
                        config.overhead_budget_pct,
                        config.region,
                        config.historical_years,
                    )
                    break

            # resolve CO2 for this step (cycle through grid data)
            co2 = float(carbon[idx])
            if not isfinite(co2):
                # TODO: maybe set a flag if we used shit data??
                co2 = mean_co2

            dt_s = step_s
            cur_time = wall_clock

            # --- handle ongoing checkpoint transition -------------------
            if transition_timer_s > 0:
                spent_s = min(dt_s, transition_timer_s)
                transition_timer_s -= spent_s

                e_wh = energy_wh(train_power_w, spent_s)
                em_g = emissions_g(e_wh, co2)
                checkpoint_s += spent_s
                checkpoint_energy_wh += e_wh
                total_energy_wh += e_wh
                total_emissions_g += em_g
                total_wall_s += spent_s
                wall_clock += timedelta(seconds=spent_s)

                if record_time_series:
                    ts_timestamps.append(cur_time)
                    ts_carbon.append(co2)
                    ts_state.append(
                        f"checkpointing -> {_state_label(target_after_transition)}"
                    )

                dt_s -= spent_s

                if transition_timer_s <= 0:
                    assert target_after_transition is not None
                    state = target_after_transition
                    target_after_transition = None

                if dt_s <= 0:
                    idx = (idx + 1) % n_points
                    continue
                cur_time = wall_clock

            # --- evaluate policy -----------------------------------------
            action = policy.evaluate(co2, state == SimState.PAUSED)

            if action == PolicyAction.PAUSE and state == SimState.RUNNING:
                num_pauses += 1
                overhead_s = profile.checkpoint_pause_time
                if overhead_s <= 0:
                    state = SimState.PAUSED
                else:
                    transition_timer_s = overhead_s
                    target_after_transition = SimState.PAUSED
                    continue

            if action == PolicyAction.RESUME and state == SimState.PAUSED:
                overhead_s = profile.checkpoint_resume_time
                if overhead_s <= 0:
                    state = SimState.RUNNING
                else:
                    transition_timer_s = overhead_s
                    target_after_transition = SimState.RUNNING
                    continue

            # --- spend dt_s in current state ----------------------------
            if state == SimState.RUNNING:
                max_tokens = int(tps * dt_s)
                tokens_step = min(max_tokens, tokens_remaining)
                effective_s = dt_s if tokens_step >= max_tokens else (tokens_step / tps)
                tokens_remaining -= tokens_step

                e_wh = energy_wh(train_power_w, effective_s)
                em_g = emissions_g(e_wh, co2)

                training_s += effective_s
                training_energy_wh += e_wh
                total_energy_wh += e_wh
                total_emissions_g += em_g
                total_wall_s += effective_s
                wall_clock += timedelta(seconds=effective_s)

                # idle the remainder of the step if tokens finished early
                idle_remainder_s = dt_s - effective_s
                if idle_remainder_s > 0:
                    idle_wh = energy_wh(pause_power_w, idle_remainder_s)
                    paused_s += idle_remainder_s
                    paused_energy_wh += idle_wh
                    total_energy_wh += idle_wh
                    total_emissions_g += emissions_g(idle_wh, co2)
                    total_wall_s += idle_remainder_s
                    wall_clock += timedelta(seconds=idle_remainder_s)
            else:
                e_wh = energy_wh(pause_power_w, dt_s)
                em_g = emissions_g(e_wh, co2)

                paused_s += dt_s
                paused_energy_wh += e_wh
                total_energy_wh += e_wh
                total_emissions_g += em_g
                total_wall_s += dt_s
                wall_clock += timedelta(seconds=dt_s)

            if record_time_series:
                ts_timestamps.append(cur_time)
                ts_carbon.append(co2)
                ts_state.append(_state_label(state))

            idx = (idx + 1) % n_points

        # ----------------------------------------------------------------
        # Assemble result
        # ----------------------------------------------------------------
        overhead_s = paused_s + checkpoint_s
        actual_overhead_pct = (
            0.0 if training_s == 0 else 100.0 * overhead_s / training_s
        )

        return SimulationResult(
            scenario_description=config.scenario_description,
            model=profile.name,
            region=config.region,
            historical_years=config.historical_years,
            start_time=start_time,
            threshold=config.theta_pause,
            hysteresis_margin=config.theta_resume,
            total_wall_time_h=total_wall_s / 3600.0,
            training_time_h=training_s / 3600.0,
            paused_time_h=paused_s / 3600.0,
            checkpoint_overhead_h=checkpoint_s / 3600.0,
            total_energy_kwh=total_energy_wh / 1000.0,
            training_energy_kwh=training_energy_wh / 1000.0,
            paused_energy_kwh=paused_energy_wh / 1000.0,
            checkpoint_energy_kwh=checkpoint_energy_wh / 1000.0,
            total_emissions_kgco2=total_emissions_g / 1000.0,
            tokens_processed=profile.dataset_tokens - tokens_remaining,
            tokens_total=profile.dataset_tokens,
            completed=(tokens_remaining <= 0),
            num_pauses=num_pauses,
            overhead_budget_pct=config.overhead_budget_pct,
            actual_overhead_pct=actual_overhead_pct,
            within_overhead_budget=(
                overhead_s / max(training_s, 1.0) <= config.overhead_budget_pct / 100.0
            ),
            timestamps=ts_timestamps,
            carbon_intensity_series=ts_carbon,
            state_series=ts_state,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_start_index(timestamps: np.ndarray, start_time: datetime) -> int:
    ts_utc = start_time.astimezone(timezone.utc).replace(tzinfo=None)
    target = np.datetime64(ts_utc)
    idx = int(np.searchsorted(timestamps, target, side="left"))
    return min(idx, len(timestamps) - 1)


def _state_label(state: SimState | None) -> str:
    if state is None:
        return "unknown"
    return state.name.lower()
