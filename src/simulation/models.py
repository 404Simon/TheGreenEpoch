"""Data models and loading code for the CO2-aware LLM training simulation."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("simulation.models")

# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_SUFFIX_MULTIPLIERS: dict[str, float] = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}


def _parse_number(raw: str) -> float:
    """Parse a CSV value that may include B/T/M/K suffixes (e.g. '14.8T')."""
    raw = raw.strip()
    if not raw:
        return 0.0
    for suffix, multiplier in _SUFFIX_MULTIPLIERS.items():
        if raw.endswith(suffix):
            return float(raw[:-1]) * multiplier
    return float(raw)


def _parse_int(raw: str) -> int:
    return int(_parse_number(raw))


def _parse_csv_list(raw: str, cast: type = str) -> list:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [cast(p) for p in parts]


def _model_name_from_stem(stem: str) -> str:
    name = re.sub(r"[_-]?[vk]\d+$", "", stem, flags=re.IGNORECASE)
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRunProfile:
    """Immutable snapshot of model architecture and hardware constants.

    Attributes
    ----------
    name : str
        Short machine-readable key (e.g. ``"Deepseek"``, ``"Kimi"``).
    model_params : int
        Number of model parameters.
    dataset_tokens : int
        Number of training tokens.
    gpu_count : int
        Number of GPUs in the training cluster.
    gpu_power_train : float
        Average active GPU power draw in W.
    gpu_power_pause : float
        Average paused/idle GPU power draw in W.
    pue : float
        Data-centre power usage effectiveness ratio.
    checkpoint_pause_time : float
        Time penalty for saving a checkpoint on pause, in seconds.
    checkpoint_resume_time : float
        Time penalty for loading a checkpoint on resume, in seconds.
    """

    name: str
    model_params: int
    dataset_tokens: int
    gpu_count: int
    gpu_power_train: float
    gpu_power_pause: float
    pue: float
    checkpoint_pause_time: float
    checkpoint_resume_time: float


@dataclass(frozen=True)
class ScenarioParameters:
    """One simulation scenario (row from scenarios.csv).

    Attributes
    ----------
    description : str
        Human-readable label.
    model : str
        Model name key matching a `TrainingRunProfile.name`.
    thresholds : list[float]
        Pause thresholds :math:`\\theta_{\\text{pause}}` in gCO₂eq/kWh.
    hysteresis : list[float]
        Hysteresis margins :math:`\\delta_{\\text{hyst}}` in gCO₂eq/kWh.
    region : str
        Grid zone code (e.g. ``"DE"``, ``"SE"``).
    start_times : list[datetime]
        Candidate training start datetimes (UTC).
    historical_years : list[int]
        Historical years for replay.
    overhead_budget_pct : float
        Maximum acceptable time overhead in percent.
    """

    description: str
    model: str
    thresholds: list[float]
    hysteresis: list[float]
    region: str
    start_times: list[datetime]
    historical_years: list[int]
    overhead_budget_pct: float

    def expand(self) -> list[SimulationConfig]:
        """Yield one ``SimulationConfig`` per threshold/start-time combo."""
        configs: list[SimulationConfig] = []
        for start in self.start_times:
            for thresh, hyst in zip(self.thresholds, self.hysteresis):
                configs.append(
                    SimulationConfig(
                        scenario_description=self.description,
                        region=self.region,
                        historical_years=self.historical_years,
                        start_time=start,
                        theta_pause=thresh,
                        theta_resume=hyst,
                        overhead_budget_pct=self.overhead_budget_pct,
                    )
                )
        return configs


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a single simulation run (one threshold pair +
    one start time)."""

    scenario_description: str
    region: str
    historical_years: list[int]
    start_time: datetime
    theta_pause: float
    theta_resume: float
    overhead_budget_pct: float
    epochs: int = 1


@dataclass
class GridData:
    """Grid carbon intensity time-series for one zone and year.

    Attributes
    ----------
    zone : str
        Grid zone code (e.g. ``"DE"``).
    year : int
        Year the data covers.
    timestamps : np.ndarray
        UTC timestamps as ``datetime64[ns]``.
    carbon_intensity : np.ndarray
        Grid carbon intensity in gCO₂eq/kWh.
    is_estimated : np.ndarray
        Boolean mask where ``True`` means value was estimated by the
        upstream Electricity Maps API.
    mean_intensity : float
        Pre-computed mean of non-estimated values; used as fallback
        when data is missing.
    """

    zone: str
    year: int
    timestamps: np.ndarray
    carbon_intensity: np.ndarray
    is_estimated: np.ndarray
    mean_intensity: float


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_training_profiles(data_dir: str | Path) -> dict[str, TrainingRunProfile]:
    """Load training profiles from ``data/*.csv``.

    Reads ``datacenter.csv`` for hardware constants and auto-discovers
    model profile CSVs (any other CSV with a ``variable`` / ``report_value``
    schema).  Returns a dict keyed by derived model name suitable for
    referencing in ``scenarios.csv``.
    """
    data_dir = Path(data_dir)

    constants: dict[str, float] = {}
    datacenter_path = data_dir / "datacenter.csv"
    if datacenter_path.exists():
        with datacenter_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                constants[row["variable"]] = _parse_number(row["report_value"])
    else:
        logger.warning("datacenter.csv not found at %s", datacenter_path)

    profiles: dict[str, TrainingRunProfile] = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name in ("datacenter.csv", "scenarios.csv"):
            continue

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            if "variable" not in fieldnames or "report_value" not in fieldnames:
                continue
            params: dict[str, str] = {}
            for row in reader:
                params[row["variable"]] = row["report_value"]

        name = _model_name_from_stem(csv_path.stem)

        profiles[name] = TrainingRunProfile(
            name=name,
            model_params=_parse_int(params.get("model_params", "0")),
            dataset_tokens=_parse_int(params.get("dataset_tokens", "0")),
            gpu_count=_parse_int(params.get("gpu_count", "0")),
            gpu_power_train=constants.get("gpu_power_train", 0.0),
            gpu_power_pause=constants.get("gpu_power_pause", 0.0),
            pue=constants.get("pue", 0.0),
            checkpoint_pause_time=constants.get("checkpoint_pause_time", 0.0),
            checkpoint_resume_time=constants.get("checkpoint_resume_time", 0.0),
        )
        logger.info("Loaded training profile: %s", name)

    if not profiles:
        logger.warning("No training profiles loaded from %s", data_dir)

    return profiles


def load_scenarios(data_dir: str | Path) -> list[ScenarioParameters]:
    """Load simulation scenarios from ``data/scenarios.csv``."""
    data_dir = Path(data_dir)
    scenarios_path = data_dir / "scenarios.csv"

    if not scenarios_path.exists():
        logger.warning("scenarios.csv not found at %s", scenarios_path)
        return []

    scenarios: list[ScenarioParameters] = []
    with scenarios_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scenarios.append(
                ScenarioParameters(
                    description=row["description"].strip(),
                    model=row["model"].strip(),
                    thresholds=_parse_csv_list(row["threshold"], float),
                    hysteresis=_parse_csv_list(row["hysteresis"], float),
                    region=row["region"].strip(),
                    start_times=[
                        datetime(1970, *map(int, ts.split("-")), tzinfo=timezone.utc)
                        for ts in _parse_csv_list(row["start_times_set"])
                    ],
                    historical_years=_parse_csv_list(row["historical"], int),
                    overhead_budget_pct=float(row["overhead_budget_pct"]),
                )
            )

    logger.info("Loaded %d scenarios", len(scenarios))
    return scenarios


def load_grid_data(data_dir: str | Path, /, zone: str, year: int) -> GridData | None:
    """Load CO₂ intensity time-series for *zone* and *year*.

    Returns ``None`` if the CSV file does not exist.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "co2_intensity" / zone / f"carbon_intensity_{year}.csv"

    if not csv_path.exists():
        logger.warning("Grid data not found: %s", csv_path)
        return None

    df = pd.read_csv(
        csv_path,
        dtype={"carbonIntensity": float, "isEstimated": bool},
        usecols=["timestamp", "carbonIntensity", "isEstimated"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    carbon_intensity = df["carbonIntensity"].to_numpy(dtype=np.float64)
    is_estimated = df["isEstimated"].to_numpy(dtype=bool)

    valid = ~is_estimated
    if valid.any():
        mean_intensity = float(carbon_intensity[valid].mean())
    else:
        mean_intensity = float(carbon_intensity.mean())

    logger.info(
        "Loaded grid data: zone=%s year=%d rows=%d mean_intensity=%.1f",
        zone,
        year,
        len(df),
        mean_intensity,
    )

    return GridData(
        zone=zone,
        year=year,
        timestamps=timestamps,
        carbon_intensity=carbon_intensity,
        is_estimated=is_estimated,
        mean_intensity=mean_intensity,
    )


def load_all_grid_data(
    data_dir: str | Path,
    /,
    *,
    zones: list[str] | None = None,
    years: list[int] | None = None,
) -> dict[tuple[str, int], GridData]:
    """Bulk-load grid data for multiple zones and years.

    If *zones* or *years* is ``None``, they are auto-discovered from the
    ``co2_intensity/`` directory tree.
    """
    data_dir = Path(data_dir)
    base = data_dir / "co2_intensity"

    if zones is None:
        zones = sorted(p.name for p in base.iterdir() if p.is_dir())

    results: dict[tuple[str, int], GridData] = {}
    for zone in zones:
        zone_dir = base / zone
        if not zone_dir.is_dir():
            continue
        if years is None:
            for csv_path in sorted(zone_dir.glob("carbon_intensity_*.csv")):
                year = int(csv_path.stem.rsplit("_", 1)[-1])
                gd = load_grid_data(data_dir, zone, year)
                if gd is not None:
                    results[(zone, year)] = gd
        else:
            for year in years:
                gd = load_grid_data(data_dir, zone, year)
                if gd is not None:
                    results[(zone, year)] = gd

    return results
