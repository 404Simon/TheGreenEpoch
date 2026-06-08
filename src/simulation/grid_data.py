"""Grid CO2 intensity data provider and CSV loading."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .models import GridData

logger = logging.getLogger("simulation.grid_data")


def load_grid_data(data_dir: Path, zone: str, year: int) -> GridData | None:
    """Load a single year of CO2 intensity CSV data into a GridData object."""
    path = data_dir / "co2_intensity" / zone / f"carbon_intensity_{year}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    timestamps = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[ns]")
    carbon_intensity = df["carbonIntensity"].to_numpy(dtype=np.float64)
    is_estimated = df["isEstimated"].to_numpy(dtype=bool)

    valid = ~is_estimated
    mean_intensity = (
        float(np.mean(carbon_intensity[valid]))
        if valid.any()
        else float(np.mean(carbon_intensity))
    )

    return GridData(
        zone=zone,
        year=year,
        timestamps=timestamps,
        carbon_intensity=carbon_intensity,
        is_estimated=is_estimated,
        mean_intensity=mean_intensity,
    )


class GridDataProvider:
    """CO2 intensity data source backed by per-year CSV files.

    The provider loads historical grid intensity series from the
    ``data/co2_intensity/<zone>/carbon_intensity_<year>.csv`` files and
    aggregates multiple years by matching the same timestamp within a year.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else None

    def _load_grid_data_for_years(self, zone: str, years: list[int]) -> list[GridData]:
        if self._data_dir is None:
            raise ValueError("GridDataProvider requires a data_dir")
        if not years:
            raise ValueError("At least one historical year must be provided")

        unique_years = sorted(set(years))
        loaded: list[GridData] = []
        missing: list[int] = []
        for year in unique_years:
            gd = load_grid_data(self._data_dir, zone, year)
            if gd is None:
                missing.append(year)
            else:
                loaded.append(gd)

        if missing:
            logger.warning("Missing grid data for %s years %s", zone, missing)
        if not loaded:
            raise ValueError(f"No grid data available for zone={zone} years={years}")
        return loaded

    @staticmethod
    def _to_canonical_year_index(timestamps: np.ndarray, target_year: int) -> pd.Index:
        ts = pd.to_datetime(timestamps)
        year_diff = target_year - ts.year[0] if len(ts) else 0
        if year_diff != 0:
            ts = ts + pd.DateOffset(years=year_diff)
        return pd.Index(ts, dtype="datetime64[ns]")

    def _data_frame_for_year(self, gd: GridData, canonical_year: int) -> pd.DataFrame:
        ts = pd.to_datetime(gd.timestamps)
        year_diff = canonical_year - gd.year
        if year_diff != 0:
            ts = ts + pd.DateOffset(years=year_diff)

        return pd.DataFrame(
            {f"carbon_{gd.year}": gd.carbon_intensity},
            index=pd.Index(ts, dtype="datetime64[ns]"),
        )

    def timeline(self, zone: str, years: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, carbon_intensity) averaged over *years*."""
        grid_data = self._load_grid_data_for_years(zone, years)
        canonical_year = min(years)

        if len(grid_data) == 1:
            return grid_data[0].timestamps, grid_data[0].carbon_intensity

        frames: list[pd.DataFrame] = []
        for gd in grid_data:
            frame = self._data_frame_for_year(gd, canonical_year)
            if frame.empty:
                continue
            frames.append(frame)

        if not frames:
            raise ValueError(
                f"Unable to align timestamps for zone={zone} years={years}"
            )

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, how="inner")

        if merged.empty:
            raise ValueError(f"No common datapoints for zone={zone} years={years}")

        merged.sort_index(inplace=True)
        timestamps = merged.index.to_numpy(dtype="datetime64[ns]")
        carbon = merged.mean(axis=1).to_numpy(dtype=np.float64)
        return timestamps, carbon

    def granularity(self, zone: str, years: list[int]) -> timedelta:
        grid_data = self._load_grid_data_for_years(zone, [years[0]])
        timestamps = grid_data[0].timestamps
        if len(timestamps) < 2:
            raise ValueError(f"Insufficient timestamps to infer granularity for {zone}")
        diff_ns = int(timestamps[1].astype("int64") - timestamps[0].astype("int64"))
        return timedelta(microseconds=diff_ns / 1000)

    def year_average(self, zone: str, years: list[int]) -> float:
        _, carbon = self.timeline(zone, years)
        if len(carbon) == 0:
            raise ValueError(
                f"No aggregated carbon intensity available for {zone} years {years}"
            )
        return float(np.mean(carbon))
