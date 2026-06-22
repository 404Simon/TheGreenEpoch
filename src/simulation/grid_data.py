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

    def _data_frame_for_year(self, gd: GridData, canonical_year: int) -> pd.DataFrame:
        ts = pd.to_datetime(gd.timestamps)
        year_diff = canonical_year - gd.year
        if year_diff != 0:
            ts = ts + pd.DateOffset(years=year_diff)

        df = pd.DataFrame(
            {f"carbon_{gd.year}": gd.carbon_intensity},
            index=pd.Index(ts, dtype="datetime64[ns]"),
        )

        # When a leap-year (e.g. 2024-02-29) shifts to a non-leap canonical
        # year, Feb 29 snaps to Feb 28, colliding with the original Feb 28
        # entries.  Group-by-index and average so that no index duplicates
        # reach the inner join in timeline() - duplicates would otherwise
        # produce a Cartesian product, double-counting leap-year data.
        if df.index.duplicated().any():
            df = df.groupby(df.index).mean()

        return df

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

        # Validate data spacing: compare first 10 intervals for consistency
        diffs_ns = timestamps[1:11].astype("int64") - timestamps[:10].astype("int64")
        if not (diffs_ns == diffs_ns[0]).all():
            logger.warning(
                "Variable data spacing in %s %s – first 10 intervals differ",
                zone,
                years[0],
            )
        diff_ns = int(diffs_ns[0])
        return timedelta(microseconds=diff_ns / 1000)

    def year_average(self, zone: str, years: list[int]) -> float:
        _, carbon = self.timeline(zone, years)
        if len(carbon) == 0:
            raise ValueError(
                f"No aggregated carbon intensity available for {zone} years {years}"
            )
        return float(np.mean(carbon))
