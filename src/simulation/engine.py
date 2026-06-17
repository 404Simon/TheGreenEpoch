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
                    "Unknown model '%s' in scenario '%s' - skipping",
                    scenario.model,
                    scenario.description,
                )
                continue

            for config in scenario.expand():
                try:
                    result = self.run_one(
                        profile, config, record_time_series=record_time_series
                    )
                except ValueError as exc:
                    logger.error(
                        "Simulation failed for %s / %s: %s",
                        config.region,
                        config.historical_years,
                        exc,
                    )
                    result = SimulationResult(
                        scenario_description=config.scenario_description,
                        model=profile.name,
                        region=config.region,
                        historical_years=config.historical_years,
                        start_time=config.start_time,
                        threshold=config.theta_pause,
                        hysteresis_margin=config.theta_resume,
                        tokens_total=profile.dataset_tokens * config.epochs,
                        issues=[f"ValueError: {exc}"],
                        stop_reason="data_error",
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
            raise ValueError(
                f"Insufficient grid data for {config.region} / {config.historical_years}"
            )

        # -- derived constants -------------------------------------------
        tps = tokens_per_second(profile.gpu_count)
        train_power_w = profile.gpu_count * profile.gpu_power_train * profile.pue
        pause_power_w = profile.gpu_count * profile.gpu_power_pause * profile.pue

        year_avg = float(
            self._provider.year_average(config.region, config.historical_years)
        )
        mean_co2 = year_avg if isfinite(year_avg) else float(np.nanmean(carbon))
        if not isfinite(mean_co2):
            mean_co2 = 0.0

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
        base_year = timestamps[0].astype("datetime64[Y]").astype(int) + 1970
        wall_clock = start_time.replace(year=int(base_year))
        idx = start_idx
        n_points = len(timestamps)

        # -- state machine -----------------------------------------------
        policy = PolicyControl(
            theta_pause=config.theta_pause, theta_resume=config.theta_resume
        )

        state = SimState.RUNNING
        transition_timer_s: float = 0.0
        target_after_transition: SimState | None = None

        tokens_remaining = profile.dataset_tokens * config.epochs

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

        # time series
        ts_timestamps: list[datetime] = []
        ts_carbon: list[float] = []
        ts_state: list[str] = []

        # diagnostics
        issues: list[str] = []
        nan_fallbacks: int = 0

        # -- resolve initial state ---------------------------------------
        init_co2 = float(carbon[start_idx])
        if not isfinite(init_co2):
            init_co2 = mean_co2
            nan_fallbacks += 1

        if init_co2 > config.theta_pause:
            ckpt_s = profile.checkpoint_pause_time
            if ckpt_s > 0:
                transition_timer_s = ckpt_s
                target_after_transition = SimState.PAUSED
                num_pauses += 1
            else:
                state = SimState.PAUSED
                num_pauses += 1

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        while tokens_remaining > 0:
            iterations += 1
            if iterations > max_iterations:
                issues.append(f"Iteration limit ({max_iterations}) reached - aborted")
                logger.warning(
                    "Iteration limit reached for %s / %s - aborting",
                    config.region,
                    config.historical_years,
                )
                break

            # budget check: stop if overhead already exceeded allowance
            if training_s > 0:
                overhead_s = paused_s + checkpoint_s
                overhead_pct = 100.0 * overhead_s / training_s
                if overhead_pct > config.overhead_budget_pct:
                    issues.append(
                        f"Overhead {overhead_pct:.1f}% exceeds budget "
                        f"{config.overhead_budget_pct:.0f}% - stopped"
                    )
                    logger.info(
                        "Overhead %.1f%% exceeds budget %.0f%% - "
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
                co2 = mean_co2
                nan_fallbacks += 1

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

        if nan_fallbacks > 0:
            issues.append(
                f"{nan_fallbacks} CO2 data point(s) were NaN/inf - "
                f"fell back to year-mean {mean_co2:.0f} gCO₂eq/kWh"
            )

        stop_reason = (
            "completed"
            if tokens_remaining <= 0
            else (
                "budget_exceeded"
                if (
                    overhead_s / max(training_s, 1.0)
                    > config.overhead_budget_pct / 100.0
                )
                else "iteration_limit"
            )
        )

        # calculate baseline emissions for training the same profile without pauses
        start_ts = np.datetime64(start_time)
        end_ts = start_ts + np.timedelta64(int(training_s), 's')

        if training_s <= 0:
            avg_co2_during_baseline_training = mean_co2
        else:
            s_idx = int(np.searchsorted(timestamps, start_ts, side="left"))
            e_idx = int(np.searchsorted(timestamps, end_ts, side="left"))
            if e_idx <= s_idx:
                avg_co2_during_baseline_training = mean_co2
            else:
                window = carbon[s_idx:e_idx]
                if window.size == 0:
                    avg_co2_during_baseline_training = mean_co2
                else:
                    avg_co2_during_baseline_training = float(np.nanmean(window))

        baseline_emissions_g = energy_wh(train_power_w, training_s) * avg_co2_during_baseline_training
        

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
            baseline_emissions_kgco2=baseline_emissions_g / 1000.0,
            tokens_processed=profile.dataset_tokens * config.epochs - tokens_remaining,
            tokens_total=profile.dataset_tokens * config.epochs,
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
            issues=issues,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_start_index(timestamps: np.ndarray, start_time: datetime) -> int:
    base_year = timestamps[0].astype("datetime64[Y]").astype(int) + 1970
    target_dt = start_time.replace(year=int(base_year))
    if target_dt.tzinfo is None:
        target_utc = target_dt
    else:
        target_utc = target_dt.astimezone(timezone.utc).replace(tzinfo=None)
    target = np.datetime64(target_utc)
    idx = int(np.searchsorted(timestamps, target, side="left"))
    return min(idx, len(timestamps) - 1)


def _state_label(state: SimState | None) -> str:
    if state is None:
        return "unknown"
    return state.name.lower()
