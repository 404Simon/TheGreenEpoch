#!/usr/bin/env python3
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import requests

API_URL = "https://api.electricitymaps.com/v3/carbon-intensity/past-range"


logger = logging.getLogger("carbon_intensity_fetcher")


def setup_logging(log_level: str = "INFO") -> None:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> Dict:
    logger.info("Loading config from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.debug("Loaded config keys: %s", sorted(config.keys()))
    return config


def parse_date(date_str: str) -> datetime:
    logger.debug("Parsing date: %s", date_str)
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def daterange(start_date: datetime, end_date: datetime) -> Iterable[datetime]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def safe_zone_name(zone: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in zone)


def get_request_span_days(temporal_granularity: str) -> int:
    """Return the number of days to fetch per request based on granularity."""
    granularity_spans = {
        "5_minutes": 1,
        "15_minutes": 3,
        "hourly": 10,
        "daily": 365,
    }
    return granularity_spans.get(temporal_granularity, 1)


def extract_rows(payload: Dict, zone: str, requested_day: str) -> List[Dict[str, str]]:
    # Handle single-point response (no history/data array)
    if "history" not in payload and "data" not in payload:
        row = {
            "zone": zone,
            "requested_day": requested_day,
            "timestamp": payload.get("datetime") or payload.get("updatedAt") or "",
            "carbonIntensity": payload.get("carbonIntensity"),
            "isEstimated": payload.get("isEstimated"),
            "estimationMethod": payload.get("estimationMethod"),
        }
        logger.info("Zone=%s day=%s single-point response: carbonIntensity=%s", 
                   zone, requested_day, row["carbonIntensity"])
        return [row]
    
    # Handle array responses
    history = payload.get("history") or payload.get("data") or []
    logger.info("Zone=%s day=%s array response contains %d entries", zone, requested_day, len(history))
    
    rows = []
    for item in history:
        rows.append({
            "zone": zone,
            "requested_day": requested_day,
            "timestamp": item.get("datetime") or item.get("updatedAt") or "",
            "carbonIntensity": item.get("carbonIntensity"),
            "isEstimated": item.get("isEstimated"),
            "estimationMethod": item.get("estimationMethod"),
        })
    return rows


def fetch_for_day(
    session: requests.Session,
    token: str,
    zone: str,
    day_start: datetime,
    day_end: datetime,
    temporal_granularity: str,
    disable_estimations: bool,
    sleep_seconds: float,
) -> List[Dict[str, str]]:
    request_datetime = iso_z(day_start)
    params = {
        "zone": zone,
        "start": iso_z(day_start),
        "end": iso_z(day_end),
        "temporalGranularity": temporal_granularity,
        "disableEstimations": str(disable_estimations).lower()
    }
    headers = {"auth-token": token}

    logger.info(
        "Requesting carbon intensity: zone=%s day=%s datetime=%s granularity=%s",
        zone,
        day_start.strftime("%Y-%m-%d"),
        request_datetime,
        temporal_granularity,
    )
    logger.debug("Request URL=%s params=%s", API_URL, params)

    response = session.get(API_URL, params=params, headers=headers, timeout=60)
    logger.info(
        "Received HTTP %s for zone=%s day=%s",
        response.status_code,
        zone,
        day_start.strftime("%Y-%m-%d"),
    )
    logger.debug("Final request URL: %s", response.url)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        logger.error(
            "HTTP error for zone=%s day=%s. Response text: %s",
            zone,
            day_start.strftime("%Y-%m-%d"),
            response.text[:2000],
        )
        raise

    payload = response.json()
    logger.debug(
        "Parsed JSON payload for zone=%s day=%s with top-level keys: %s",
        zone,
        day_start.strftime("%Y-%m-%d"),
        sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
    )

    if sleep_seconds > 0:
        logger.debug("Sleeping %.3f seconds before next request", sleep_seconds)
        time.sleep(sleep_seconds)

    return extract_rows(payload, zone, day_start.strftime("%Y-%m-%d"))


def write_csv(output_dir: Path, zone: str, rows: List[Dict[str, str]], year: int = None) -> Path:
    # Create zone subdirectory
    zone_dir = output_dir / zone
    zone_dir.mkdir(parents=True, exist_ok=True)
    
    if year:
        output_path = zone_dir / f"carbon_intensity_{year}.csv"
    else:
        output_path = zone_dir / f"carbon_intensity.csv"
    fieldnames = ["zone", "requested_day", "timestamp", "carbonIntensity", "isEstimated", "estimationMethod"]

    logger.info("Writing %d rows to %s", len(rows), output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Finished writing CSV for zone=%s: %s", zone, output_path)
    return output_path


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"

    pre_config_level = "INFO"
    if len(sys.argv) > 2:
        pre_config_level = sys.argv[2]
    setup_logging(pre_config_level)

    config = load_config(config_path)

    configured_log_level = config.get("log_level", pre_config_level)
    if configured_log_level.upper() != pre_config_level.upper():
        setup_logging(configured_log_level)
        logger.info("Adjusted log level from config to %s", configured_log_level.upper())

    token = config["api_token"]
    zones = config["zones"]
    start_date = parse_date(config["start_date"])
    end_date = parse_date(config["end_date"])
    temporal_granularity = config.get("temporal_granularity", "5_minutes")
    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate sleep seconds based on rate limit
    sleep_seconds = float(config.get("sleep_seconds_between_requests", 0.0))
    max_requests_per_minute = config.get("max_requests_per_minute", None)
    
    if max_requests_per_minute:
        sleep_seconds = 60.0 / max_requests_per_minute
        logger.info("Enforcing max_requests_per_minute=%d with sleep_seconds=%.3f", 
                    max_requests_per_minute, sleep_seconds)

    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    session = requests.Session()
    request_span_days = get_request_span_days(temporal_granularity)
    logger.info("Using request span of %d day(s) for granularity=%s", request_span_days, temporal_granularity)
    
    for zone in zones:
        all_rows = []
        request_count = 0
        current_date = start_date
        
        while current_date <= end_date:
            request_count += 1
            # Calculate span end, but don't exceed end_date
            span_end = current_date + timedelta(days=request_span_days - 1)
            if span_end > end_date:
                span_end = end_date
            
            span_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            span_end_date = span_end + timedelta(days=1)
            
            daily_rows = fetch_for_day(
                session, token, zone, span_start, span_end_date, 
                temporal_granularity, config.get("disable_estimations", False), sleep_seconds
            )
            all_rows.extend(daily_rows)
            
            current_date = span_end + timedelta(days=1)

        logger.info("Completed zone=%s with %d requests and %d total rows", zone, request_count, len(all_rows))
        
        # Group rows by year if spanning multiple years
        rows_by_year = {}
        for row in all_rows:
            if row["requested_day"]:
                year = int(row["requested_day"][:4])
                if year not in rows_by_year:
                    rows_by_year[year] = []
                rows_by_year[year].append(row)
        
        # Write separate files for each year if multiple years, otherwise single file
        if len(rows_by_year) > 1:
            for year in sorted(rows_by_year.keys()):
                path = write_csv(output_dir, zone, rows_by_year[year], year=year)
                print(path)
        else:
            path = write_csv(output_dir, zone, all_rows)
            print(path)

    logger.info("All zones processed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())