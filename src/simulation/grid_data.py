"""Grid CO2 intensity data provider (dummy stub — not implemented yet)."""

from datetime import datetime, timedelta

import numpy as np

# Dummy mean intensities per zone (gCO2eq/kWh)
_DUMMY_MEANS: dict[str, float] = {
    "DE": 340,
    "SE": 21,
    "FR": 29,
    "IT": 280,
    "ES": 135,
    "CN": 520,
    "US": 380,
}

_GRANULARITY = timedelta(minutes=5)
_POINTS_PER_YEAR = 105120


class GridDataProvider:
    """Dummy CO2 intensity data source — returns synthetic values.

    TODO: replace with real data loading via models.py.  The real
    implementation will load per-year CSV files, align timestamps, and
    average CO2 intensity across the requested *years* list.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        pass

    def timeline(
        self, zone: str, years: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, carbon_intensity) averaged over *years*."""
        base = datetime(years[0], 1, 1)
        timestamps = np.array(
            [base + i * _GRANULARITY for i in range(_POINTS_PER_YEAR)],
            dtype="datetime64[ns]",
        )
        mean = _DUMMY_MEANS.get(zone, 300)
        seed = abs(hash((zone, tuple(years)))) % (2**31)
        rng = np.random.default_rng(seed)
        carbon = (
            rng.normal(mean, mean * 0.3, _POINTS_PER_YEAR)
            .clip(0)
            .astype(np.float64)
        )
        return timestamps, carbon

    def granularity(
        self, zone: str, years: list[int]
    ) -> timedelta:
        return _GRANULARITY

    def year_average(self, zone: str, years: list[int]) -> float:
        return _DUMMY_MEANS.get(zone, 300)
