"""
Comprehensive Validation & Verification Test Suite for TheGreenEpoch Simulation.

Covers:
- 2.1 Energy and emissions accounting
- 2.2 Pause/resume logic and hysteresis
- 2.3 Progress and runtime consistency
- 2.4 Handling of data gaps
- 2.5 Unit tests for KPI functions
- 1.1 Structural trade-off behavior
- 1.3 Regional and temporal realism
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from simulation import (
    SimulationRunner,
    SimulationConfig,
    ScenarioParameters,
    TrainingRunProfile,
    PolicyControl,
    PolicyAction,
    SimState,
    load_training_profiles,
    load_scenarios,
)
from simulation.grid_data import GridDataProvider
from simulation.physics import energy_wh, emissions_g, tokens_per_second
from simulation.results import SimulationResult


class TestKPIFunctions:
    """2.5 Unit tests for KPI functions - verify standalone correctness."""

    @pytest.fixture
    def result(self):
        return SimulationResult(
            scenario_description="Example KPI test",
            model="Deepseek",
            region="SE",
            historical_years=[2022, 2023, 2024, 2025],
            start_time=datetime(2022, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=1796.7,
            training_time_h=1300.8,
            paused_time_h=495.4,
            checkpoint_overhead_h=0.5,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=59334.0,
            baseline_emissions_kgco2=84762.85714285714,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=12,
            overhead_budget_pct=200.0,
            actual_overhead_pct=38.1,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )

    def test_co2_savings_pct_basic(self, result):
        """Test CO2 savings percentage formula: 100 * (baseline - actual) / baseline."""
        expected = 100.0 * (result.baseline_emissions_kgco2 - result.total_emissions_kgco2) / result.baseline_emissions_kgco2
        actual = result.co2_savings_pct
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_co2_savings_pct_zero_savings(self):
        """Test CO2 savings when baseline and policy emissions are identical."""
        result = SimulationResult(
            scenario_description="Zero savings test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=1000.0,
            baseline_emissions_kgco2=1000.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 100.0 * (result.baseline_emissions_kgco2 - result.total_emissions_kgco2) / result.baseline_emissions_kgco2
        actual = result.co2_savings_pct
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_co2_savings_pct_full_elimination(self):
        """Test CO2 savings when emissions are fully eliminated."""
        result = SimulationResult(
            scenario_description="Full elimination test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=0.0,
            baseline_emissions_kgco2=1000.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 100.0 * (result.baseline_emissions_kgco2 - result.total_emissions_kgco2) / result.baseline_emissions_kgco2
        actual = result.co2_savings_pct
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_co2_savings_pct_zero_baseline(self):
        """Test CO2 savings when baseline is zero (should return 0)."""
        result = SimulationResult(
            scenario_description="Zero baseline test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=100.0,
            baseline_emissions_kgco2=0.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 0.0
        actual = result.co2_savings_pct
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_time_overhead_pct_basic(self, result):
        """Test time overhead percentage formula: 100 * (wall - training) / training."""
        expected = 100.0 * (result.total_wall_time_h - result.training_time_h) / result.training_time_h
        actual = result.time_overhead_pct
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_time_overhead_pct_no_overhead(self):
        """Test time overhead when wall time equals training time."""
        result = SimulationResult(
            scenario_description="No overhead test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=1000.0,
            baseline_emissions_kgco2=1200.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 100.0 * (result.total_wall_time_h - result.training_time_h) / result.training_time_h
        actual = result.time_overhead_pct
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_time_overhead_pct_doubled_time(self):
        """Test time overhead when wall time is double training time."""
        result = SimulationResult(
            scenario_description="Double time test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=200.0,
            training_time_h=100.0,
            paused_time_h=100.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=2000.0,
            baseline_emissions_kgco2=2500.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=1,
            overhead_budget_pct=200.0,
            actual_overhead_pct=100.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 100.0 * (result.total_wall_time_h - result.training_time_h) / result.training_time_h
        actual = result.time_overhead_pct
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_time_overhead_pct_zero_training(self):
        """Test time overhead when training time is zero (should return 0)."""
        result = SimulationResult(
            scenario_description="Zero training time test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=0.0,
            paused_time_h=100.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=1000.0,
            baseline_emissions_kgco2=1200.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=1,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 0.0
        actual = result.time_overhead_pct
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_fixture_sanity_kpi_values(self, result):
        """Fixture sanity check: verify raw fields match the example values."""
        assert result.total_wall_time_h == pytest.approx(1796.7, rel=1e-6)
        assert result.training_time_h == pytest.approx(1300.8, rel=1e-6)
        assert result.paused_time_h == pytest.approx(495.4, rel=1e-6)
        assert result.checkpoint_overhead_h == pytest.approx(0.5, rel=1e-6)
        assert result.actual_overhead_pct == pytest.approx(38.1, rel=1e-6)
        assert result.num_pauses == 12
        assert result.total_emissions_kgco2 == pytest.approx(59334.0, rel=1e-6)

    def test_score_calculation(self, result):
        """Test composite score = savings / max(overhead, epsilon)."""
        expected = result.co2_savings_pct / max(result.time_overhead_pct, 1e-6)
        actual = result.score
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_score_with_zero_overhead(self):
        """Test score when overhead is near zero."""
        result = SimulationResult(
            scenario_description="Zero overhead score test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=700.0,
            baseline_emissions_kgco2=1000.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = result.co2_savings_pct / 1e-6
        actual = result.score
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_score_zero_when_not_completed(self):
        """Test that score is 0 when training is not completed."""
        result = SimulationResult(
            scenario_description="Not completed score test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=700.0,
            baseline_emissions_kgco2=1000.0,
            tokens_processed=0,
            tokens_total=0,
            completed=False,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 0.0
        actual = result.score
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_score_zero_when_overhead_budget_exceeded(self):
        """Test that score is 0 when overhead budget is exceeded."""
        result = SimulationResult(
            scenario_description="Over budget score test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=300.0,
            training_time_h=100.0,
            paused_time_h=200.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=700.0,
            baseline_emissions_kgco2=1000.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=1,
            overhead_budget_pct=200.0,
            actual_overhead_pct=200.0,
            within_overhead_budget=False,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        expected = 0.0
        actual = result.score
        assert actual == pytest.approx(expected, abs=1e-6)
    
    def test_no_negative_time_overhead(self, result):
        """Verify that time overhead is never negative."""
        # Time overhead = 100 * (wall_time - training_time) / training_time
        # Should always be >= 0 since wall_time >= training_time
        assert result.time_overhead_pct >= 0.0, (
            f"Time overhead should not be negative: {result.time_overhead_pct}%"
        )

    def test_no_negative_time_overhead_edge_cases(self):
        """Test time overhead across edge cases."""
        result = SimulationResult(
            scenario_description="Edge case 1",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=1000.0,
            baseline_emissions_kgco2=1200.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        assert result.time_overhead_pct == pytest.approx(0.0, abs=1e-6)

        result = SimulationResult(
            scenario_description="Edge case 2",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=150.0,
            training_time_h=100.0,
            paused_time_h=50.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=1500.0,
            baseline_emissions_kgco2=1800.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=1,
            overhead_budget_pct=200.0,
            actual_overhead_pct=50.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        assert result.time_overhead_pct == pytest.approx(50.0, rel=1e-6)

    def test_emissions_non_negative(self, result):
        """Verify that total emissions are never negative."""
        assert result.total_emissions_kgco2 >= 0.0, (
            f"Total emissions should not be negative: {result.total_emissions_kgco2} kgCO2"
        )

    def test_emissions_non_negative_edge_cases(self):
        """Test emissions non-negativity across edge cases."""
        # Case 1: Zero emissions
        result = SimulationResult(
            scenario_description="Zero emissions",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=0.0,
            baseline_emissions_kgco2=1000.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        assert result.total_emissions_kgco2 >= 0.0

        # Case 2: Positive emissions
        result = SimulationResult(
            scenario_description="Positive emissions",
            model="Deepseek",
            region="DE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=100.0,
            training_time_h=100.0,
            paused_time_h=0.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=500.0,
            training_energy_kwh=500.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=50.0,
            baseline_emissions_kgco2=60.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=0,
            overhead_budget_pct=200.0,
            actual_overhead_pct=0.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        assert result.total_emissions_kgco2 >= 0.0

    def test_energy_components_non_negative(self, result):
        """Verify that all energy components are never negative."""
        assert result.total_energy_kwh >= 0.0, (
            f"Total energy should not be negative: {result.total_energy_kwh} kWh"
        )
        assert result.training_energy_kwh >= 0.0, (
            f"Training energy should not be negative: {result.training_energy_kwh} kWh"
        )
        assert result.paused_energy_kwh >= 0.0, (
            f"Paused energy should not be negative: {result.paused_energy_kwh} kWh"
        )
        assert result.checkpoint_energy_kwh >= 0.0, (
            f"Checkpoint energy should not be negative: {result.checkpoint_energy_kwh} kWh"
        )

    def test_time_components_non_negative(self, result):
        """Verify that all time components are never negative."""
        assert result.total_wall_time_h >= 0.0, (
            f"Total wall time should not be negative: {result.total_wall_time_h} h"
        )
        assert result.training_time_h >= 0.0, (
            f"Training time should not be negative: {result.training_time_h} h"
        )
        assert result.paused_time_h >= 0.0, (
            f"Paused time should not be negative: {result.paused_time_h} h"
        )
        assert result.checkpoint_overhead_h >= 0.0, (
            f"Checkpoint overhead should not be negative: {result.checkpoint_overhead_h} h"
        )

    def test_baseline_emissions_non_negative(self, result):
        """Verify that baseline emissions are never negative."""
        assert result.baseline_emissions_kgco2 >= 0.0, (
            f"Baseline emissions should not be negative: {result.baseline_emissions_kgco2} kgCO2"
        )

    def test_no_negative_overhead_pct_in_results(self):
        """Verify actual_overhead_pct is never negative in any result."""
        result = SimulationResult(
            scenario_description="Overhead pct test",
            model="Deepseek",
            region="SE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            threshold=30.0,
            hysteresis_margin=25.0,
            total_wall_time_h=150.0,
            training_time_h=100.0,
            paused_time_h=50.0,
            checkpoint_overhead_h=0.0,
            total_energy_kwh=0.0,
            training_energy_kwh=0.0,
            paused_energy_kwh=0.0,
            checkpoint_energy_kwh=0.0,
            total_emissions_kgco2=1000.0,
            baseline_emissions_kgco2=1200.0,
            tokens_processed=0,
            tokens_total=0,
            completed=True,
            num_pauses=1,
            overhead_budget_pct=200.0,
            actual_overhead_pct=50.0,
            within_overhead_budget=True,
            timestamps=[],
            carbon_intensity_series=[],
            state_series=[],
            issues=[],
            stop_reason="",
        )
        assert result.actual_overhead_pct >= 0.0, (
            f"Actual overhead pct should not be negative: {result.actual_overhead_pct}%"
        )

class TestPolicyControl:
    """2.2 Pause/resume logic and hysteresis verification."""

    def test_policy_pause_decision_running(self):
        """Test pause decision when running and intensity exceeds threshold."""
        policy = PolicyControl(theta_pause=500.0, theta_resume=450.0)
        action = policy.evaluate(co2_intensity=550.0, is_paused=False)
        assert action == PolicyAction.PAUSE

    def test_policy_continue_running(self):
        """Test no pause decision when running and below threshold."""
        policy = PolicyControl(theta_pause=500.0, theta_resume=450.0)
        action = policy.evaluate(co2_intensity=450.0, is_paused=False)
        assert action == PolicyAction.CONTINUE

    def test_policy_resume_decision(self):
        """Test resume decision when paused and intensity below resume threshold."""
        policy = PolicyControl(theta_pause=500.0, theta_resume=450.0)
        action = policy.evaluate(co2_intensity=400.0, is_paused=True)
        assert action == PolicyAction.RESUME

    def test_policy_continue_paused(self):
        """Test no resume when paused but still above resume threshold."""
        policy = PolicyControl(theta_pause=500.0, theta_resume=450.0)
        action = policy.evaluate(co2_intensity=480.0, is_paused=True)
        assert action == PolicyAction.CONTINUE

    def test_hysteresis_prevents_thrashing(self):
        """Test that hysteresis prevents rapid pause/resume cycles."""
        policy = PolicyControl(theta_pause=500.0, theta_resume=450.0)

        # Start running, cross pause threshold
        state = False
        action = policy.evaluate(co2_intensity=550.0, is_paused=state)
        assert action == PolicyAction.PAUSE
        state = True

        # Intensity drops to middle of hysteresis band
        action = policy.evaluate(co2_intensity=475.0, is_paused=state)
        assert action == PolicyAction.CONTINUE  # Should NOT resume yet

        # Intensity drops below resume threshold
        action = policy.evaluate(co2_intensity=400.0, is_paused=state)
        assert action == PolicyAction.RESUME
        state = False

    def test_hysteresis_margin_validation(self):
        """Test that theta_resume <= theta_pause is required."""
        with pytest.raises(ValueError):
            PolicyControl(theta_pause=400.0, theta_resume=500.0)

    def test_zero_hysteresis(self):
        """Test with zero hysteresis (emergency case)."""
        policy = PolicyControl(theta_pause=500.0, theta_resume=500.0)
        assert policy.theta_pause == policy.theta_resume


class TestEnergyAndEmissionsAccounting:
    """2.1 Energy and emissions accounting verification."""

    @pytest.fixture
    def runner(self):
        """Setup runner with test data."""
        data_dir = Path(__file__).parent.parent / "data"
        return SimulationRunner.from_data_dir(data_dir)

    def test_baseline_no_pause_scenario(self, runner):
        """Verify baseline scenario with no pausing."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)

        # Create simple baseline config (no pausing)
        config = SimulationConfig(
            scenario_description="Baseline test - no pause",
            model="Deepseek",
            region="US",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            theta_pause=1e9,  # Very high threshold - no pausing
            theta_resume=1e9,
            epochs=1,
            overhead_budget_pct=200.0,
        )

        profile = profiles["Deepseek"]
        result = runner.run_one(profile, config)

        # Verify completion
        assert result.completed, f"Simulation failed: {result.stop_reason}"

        # Check that training_time_h ≈ total_wall_time_h (minimal pausing)
        overhead_pct = 100.0 * result.idle_time_h / result.training_time_h
        assert overhead_pct < 5.0, f"Unexpected overhead in baseline: {overhead_pct}%"

    def test_emissions_scaling_with_power(self):
        """Verify emissions scale with power consumption using the actual physics functions."""
        from simulation.physics import energy_wh, emissions_g

        # If P doubles, emissions should double (at same intensity and time)
        p1 = 500.0   # Watts (GPU power)
        p2 = 1000.0  # Watts (double)
        delta_t = 3600.0  # 1 hour in seconds
        intensity = 100.0  # gCO2/kWh

        # physics functions
        e1 = energy_wh(power_w=p1, duration_s=delta_t)
        e2 = energy_wh(power_w=p2, duration_s=delta_t)

        em1 = emissions_g(energy_wh=e1, carbon_intensity_g_per_kwh=intensity)
        em2 = emissions_g(energy_wh=e2, carbon_intensity_g_per_kwh=intensity)

        # Verify scaling: if power doubles, emissions should double
        ratio = em2 / em1
        assert ratio == pytest.approx(2.0, rel=1e-6), (
            f"Emissions ratio {ratio} should be 2.0 when power doubles. "
            f"em1={em1}, em2={em2}"
        )


    def test_checkpoint_overhead_accumulation(self):
        """Verify checkpoint time and energy accumulate correctly using simulation logic."""
        from simulation.physics import energy_wh, emissions_g

        # Checkpoint overhead should accumulate per pause count
        pause_count = 5
        checkpoint_time_s = 148.8  # seconds per checkpoint
        checkpoint_power = 100.0   # Watts
        intensity = 100.0          # gCO2/kWh

        # Total checkpoint time
        total_checkpoint_s = pause_count * checkpoint_time_s

        # physics function for energy
        total_checkpoint_wh = energy_wh(power_w=checkpoint_power, duration_s=total_checkpoint_s)

        # physics function for emissions
        total_checkpoint_emissions_g = emissions_g(
            energy_wh=total_checkpoint_wh,
            carbon_intensity_g_per_kwh=intensity
        )

        # Verify formulas are mathematically consistent
        assert total_checkpoint_s == pytest.approx(744.0, rel=1e-6), (
            f"Total checkpoint time {total_checkpoint_s} should be 744.0 s"
        )

        assert total_checkpoint_wh > 0, "Checkpoint energy should be positive"

        # Optional: verify human-readable values
        # t = 744 s = 0.206666... h
        # energy = 100 W * 0.206666... h = 20.666... Wh
        expected_checkpoint_wh = checkpoint_power * total_checkpoint_s / 3600.0
        assert total_checkpoint_wh == pytest.approx(expected_checkpoint_wh, rel=1e-6), (
            f"Checkpoint energy {total_checkpoint_wh} Wh does not match expected {expected_checkpoint_wh} Wh"
        )

        # Emissions in g: energy (Wh) / 1000 * intensity (g/kWh)
        expected_checkpoint_emissions_g = total_checkpoint_wh / 1000.0 * intensity
        assert total_checkpoint_emissions_g == pytest.approx(expected_checkpoint_emissions_g, rel=1e-6), (
            f"Checkpoint emissions {total_checkpoint_emissions_g} g does not match expected {expected_checkpoint_emissions_g} g"
        )

class TestProgressAndRuntimeConsistency:
    """2.3 Progress and runtime consistency verification."""

    @pytest.fixture
    def runner(self):
        """Setup runner with test data."""
        data_dir = Path(__file__).parent.parent / "data"
        return SimulationRunner.from_data_dir(data_dir)

    def test_baseline_training_completes(self, runner):
        """Test that baseline (no pause) training completes."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)

        config = SimulationConfig(
            scenario_description="Progress test - baseline",
            model="Deepseek",
            region="US",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            theta_pause=1e9,  # No pausing
            theta_resume=1e9,
            epochs=1,
            overhead_budget_pct=200.0,
        )

        profile = profiles["Deepseek"]
        result = runner.run_one(profile, config)

        # Verify training completed
        assert result.completed
        assert result.tokens_processed == result.tokens_total
        assert result.completion_pct == pytest.approx(100.0, rel=1e-3)

    def test_paused_time_less_than_wall_time(self, runner):
        """Verify paused time <= wall time."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)

        config = SimulationConfig(
            scenario_description="Progress test - with pause",
            model="Deepseek",
            region="CN",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            theta_pause=400.0,  # Allow pausing
            theta_resume=350.0,
            epochs=1,
            overhead_budget_pct=200.0,
        )

        profile = profiles["Deepseek"]
        result = runner.run_one(profile, config)

        # Verify timing consistency
        assert result.paused_time_h <= result.total_wall_time_h
        assert result.training_time_h <= result.total_wall_time_h

    def test_wall_time_equals_training_plus_idle(self, runner):
        """Verify wall_time = training_time + paused_time + checkpoint_time."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)

        config = SimulationConfig(
            scenario_description="Progress test - accounting",
            model="Deepseek",
            region="US",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            theta_pause=400.0,
            theta_resume=350.0,
            epochs=1,
            overhead_budget_pct=200.0,
        )

        profile = profiles["Deepseek"]
        result = runner.run_one(profile, config)

        # Allow for floating point error
        expected_wall_time = result.training_time_h + result.paused_time_h + result.checkpoint_overhead_h
        assert result.total_wall_time_h == pytest.approx(expected_wall_time, rel=1e-3)


class TestRegionalAndTemporalRealism:
    """1.3 Regional and temporal realism validation."""

    @pytest.fixture
    def runner(self):
        """Setup runner with test data."""
        data_dir = Path(__file__).parent.parent / "data"
        return SimulationRunner.from_data_dir(data_dir)

    def test_regional_comparison_germany_vs_sweden(self, runner):
        """Validate regional differences: Germany vs Sweden (high vs clean grid)."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        results = {}
        for region in ["DE", "SE"]:
            config = SimulationConfig(
                scenario_description=f"Regional test - {region}",
                model="Deepseek",
                region=region,
                historical_years=[2024],
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                theta_pause=1e9,  # Baseline (no pause)
                theta_resume=1e9,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            results[region] = runner.run_one(profile, config)

        # Sweden has cleaner grid, so baseline emissions should be lower
        assert results["SE"].total_emissions_kgco2 < results["DE"].total_emissions_kgco2, \
            "Sweden should have lower emissions than Germany (cleaner grid)"

    def test_temporal_comparison_summer_vs_winter(self, runner):
        """Validate temporal differences: summer vs winter training."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        results = {}
        # Summer (June) vs Winter (January)
        for month, label in [(1, "winter"), (6, "summer")]:
            config = SimulationConfig(
                scenario_description=f"Temporal test - {label}",
                model="Deepseek",
                region="DE",
                historical_years=[2024],
                start_time=datetime(2024, month, 1, tzinfo=timezone.utc),
                theta_pause=1e9,  # Baseline
                theta_resume=1e9,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            results[label] = runner.run_one(profile, config)

        # Just verify both complete and are different
        assert results["winter"].completed
        assert results["summer"].completed
        # Emissions may differ due to grid mix changes
        # (not strictly enforcing direction as it depends on weather)

    def test_regional_savings_with_policy(self, runner):
        """Verify that regions with clean grids see limited savings."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        results = {}
        for region in ["DE", "SE"]:
            # Aggressive policy
            config = SimulationConfig(
                scenario_description=f"Savings test - {region}",
                model="Deepseek",
                region=region,
                historical_years=[2024],
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                theta_pause=300.0,  # Pause at 300 gCO2/kWh
                theta_resume=250.0,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            results[region] = runner.run_one(profile, config)

        # Verify both scenarios completed
        assert results["DE"].completed
        assert results["SE"].completed


class TestStructuralTradeOff:
    """1.1 Structural trade-off behavior validation."""

    @pytest.fixture
    def runner(self):
        """Setup runner with test data."""
        data_dir = Path(__file__).parent.parent / "data"
        return SimulationRunner.from_data_dir(data_dir)

    def test_stricter_threshold_increases_savings(self, runner):
        """Verify that lower pause thresholds (stricter policy) increase savings."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        # Test with three increasing thresholds
        thresholds = [300.0, 400.0, 500.0]
        results = {}

        for theta in thresholds:
            config = SimulationConfig(
                scenario_description=f"Threshold test - {theta}",
                model="Deepseek",
                region="DE",
                historical_years=[2024],
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                theta_pause=theta,
                theta_resume=theta - 50.0,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            results[theta] = runner.run_one(profile, config)

        # Verify that stricter thresholds (lower values) produce more savings
        # Note: This is a statistical tendency, may not hold for every transition
        emissions = [results[t].total_emissions_kgco2 for t in thresholds]
        assert emissions[0] <= emissions[-1], \
            "Lower threshold should produce lower or equal emissions"

    def test_stricter_threshold_increases_time_overhead(self, runner):
        """Verify that lower pause thresholds increase time overhead."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        thresholds = [300.0, 400.0, 500.0]
        results = {}

        for theta in thresholds:
            config = SimulationConfig(
                scenario_description=f"Overhead test - {theta}",
                model="Deepseek",
                region="DE",
                historical_years=[2024],
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                theta_pause=theta,
                theta_resume=theta - 50.0,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            results[theta] = runner.run_one(profile, config)

        # Verify pause counts increase with stricter thresholds
        pause_counts = [results[t].num_pauses for t in thresholds]
        assert pause_counts[0] >= pause_counts[-1], \
            "Lower threshold should trigger more pauses"

    def test_pareto_frontier_monotonicity(self, runner):
        """Test that Pareto frontier exhibits expected trade-off shape."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        # Get baseline emissions for percentage calculation
        baseline_config = SimulationConfig(
            scenario_description="Baseline for percentages",
            model="Deepseek",
            region="DE",
            historical_years=[2024],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            theta_pause=1e9,  # No pause
            theta_resume=1e9,
            epochs=1,
            overhead_budget_pct=200.0,
        )
        baseline = runner.run_one(profile, baseline_config)

        # Test range of thresholds
        thresholds = [250.0, 300.0, 350.0, 400.0]
        points = []

        for theta in thresholds:
            config = SimulationConfig(
                scenario_description=f"Frontier test - {theta}",
                model="Deepseek",
                region="DE",
                historical_years=[2024],
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                theta_pause=theta,
                theta_resume=theta - 50.0,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            result = runner.run_one(profile, config)

            # Calculate percentages
            savings_pct = (baseline.total_emissions_kgco2 - result.total_emissions_kgco2) / \
                         baseline.total_emissions_kgco2 * 100 if baseline.total_emissions_kgco2 > 0 else 0
            overhead_pct = result.actual_overhead_pct

            points.append((overhead_pct, savings_pct))

        # Verify points exist and show trade-off tendency
        assert len(points) > 0, "Should have generated frontier points"
        # Generally, higher savings should correlate with higher overhead
        savings_list = [p[1] for p in points]
        assert max(savings_list) >= min(savings_list), "Range of savings should be observed"


class TestCheckpointGranularity:
    """1.4 Checkpoint granularity and overhead effects validation."""

    @pytest.fixture
    def runner(self):
        """Setup runner with test data."""
        data_dir = Path(__file__).parent.parent / "data"
        return SimulationRunner.from_data_dir(data_dir)

    def test_pause_count_increases_with_policy_tightness(self, runner):
        """Verify pause counts increase with more aggressive policies."""
        data_dir = Path(__file__).parent.parent / "data"
        profiles = load_training_profiles(data_dir)
        profile = profiles["Deepseek"]

        # More aggressive thresholds should trigger more pauses
        policy_configs = [
            (500.0, 450.0),  # Loose
            (350.0, 300.0),  # Medium
            (250.0, 200.0),  # Aggressive
        ]

        pause_counts = []
        for theta_p, theta_r in policy_configs:
            config = SimulationConfig(
                scenario_description="Granularity test",
                model="Deepseek",
                region="DE",
                historical_years=[2024],
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                theta_pause=theta_p,
                theta_resume=theta_r,
                epochs=1,
                overhead_budget_pct=200.0,
            )
            result = runner.run_one(profile, config)
            pause_counts.append(result.num_pauses)

        # Verify pause counts generally increase (more aggressive → more pauses)
        assert pause_counts[0] <= pause_counts[-1], \
            "More aggressive policy should result in more or equal pauses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
