"""Central configuration for the Open-Meteo -> Parquet pipeline.

Everything that is shared between the extract, transform, load and analytics
stages lives here: API endpoints, the city catalogue, the canonical column
schema and the on-disk layout. Keeping a single source of truth means a schema
change only has to be made in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import polars as pl

# --------------------------------------------------------------------------------------
# Paths and endpoints
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATASET_HOURLY: Final[str] = "hourly"
DATASET_ANALYTICS: Final[str] = "analytics"
SUMMARY_FILENAME: Final[str] = "summary.parquet"

FORECAST_ENDPOINT: Final[str] = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_ENDPOINT: Final[str] = "https://archive-api.open-meteo.com/v1/archive"

#: The ERA5 archive lags real time; anything fresher must come from the forecast
#: endpoint using its past_days window (which reports measured values for past hours).
ARCHIVE_LAG_DAYS: Final[int] = 5
MAX_PAST_DAYS: Final[int] = 92

RECORD_TYPE_FORECAST: Final[str] = "forecast"
RECORD_TYPE_OBSERVATION: Final[str] = "observation"
RECORD_TYPES: Final[tuple[str, ...]] = (RECORD_TYPE_FORECAST, RECORD_TYPE_OBSERVATION)

HOURLY_VARIABLES: Final[tuple[str, ...]] = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)


def default_data_dir() -> Path:
    """Return the root data directory, honouring the WEATHER_ETL_DATA_DIR override."""
    override = os.environ.get("WEATHER_ETL_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "data"


DEFAULT_DATA_DIR: Final[Path] = default_data_dir()


# --------------------------------------------------------------------------------------
# City catalogue
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class City:
    """A weather location understood by the pipeline."""

    name: str
    slug: str
    latitude: float
    longitude: float
    timezone: str = "UTC"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.name} ({self.latitude:.4f}, {self.longitude:.4f})"


CITIES: Final[tuple[City, ...]] = (
    City("London", "london", 51.5074, -0.1278, "Europe/London"),
    City("Tokyo", "tokyo", 35.6762, 139.6503, "Asia/Tokyo"),
    City("New York", "new_york", 40.7128, -74.0060, "America/New_York"),
)

CITY_SLUGS: Final[tuple[str, ...]] = tuple(city.slug for city in CITIES)


class ConfigurationError(ValueError):
    """Raised when the caller asks for an unknown city, dataset or date range."""


def get_city(slug: str) -> City:
    """Return the :class:`City` registered under `slug` (case insensitive)."""
    needle = slug.strip().lower()
    for city in CITIES:
        if city.slug == needle or city.name.lower() == needle:
            return city
    known = ", ".join(CITY_SLUGS)
    raise ConfigurationError(f"Unknown city {slug!r}. Known cities: {known}")


def resolve_cities(slugs: "list[str] | tuple[str, ...] | None") -> tuple[City, ...]:
    """Resolve a user supplied list of city slugs, defaulting to the full catalogue."""
    if not slugs:
        return CITIES
    resolved: list[City] = []
    seen: set[str] = set()
    for slug in slugs:
        city = get_city(slug)
        if city.slug not in seen:
            seen.add(city.slug)
            resolved.append(city)
    return tuple(resolved)


def utc_today() -> date:
    """Return the current UTC date (patched in tests for determinism)."""
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).date()


def archive_cutoff(today: date | None = None) -> date:
    """Return the newest date that the archive API is expected to serve."""
    return (today or utc_today()) - timedelta(days=ARCHIVE_LAG_DAYS)


# --------------------------------------------------------------------------------------
# Canonical schema
# --------------------------------------------------------------------------------------

#: Canonical column order and dtypes for every hourly Parquet file.
#: `year` / `month` / `day` are intentionally absent: those are encoded in the
#: directory names and surfaced by DuckDB hive partitioning, so duplicating them
#: in the file body would only invite drift.
HOURLY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "city": pl.String,
    "city_slug": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "record_type": pl.String,
    "source_endpoint": pl.String,
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "date_utc": pl.Date,
    "hour_utc": pl.Int8,
    "temperature_2m": pl.Float64,
    "relative_humidity_2m": pl.Float64,
    "apparent_temperature": pl.Float64,
    "precipitation": pl.Float64,
    "rain": pl.Float64,
    "snowfall": pl.Float64,
    "weather_code": pl.Int16,
    "cloud_cover": pl.Float64,
    "pressure_msl": pl.Float64,
    "surface_pressure": pl.Float64,
    "wind_speed_10m": pl.Float64,
    "wind_direction_10m": pl.Float64,
    "wind_gusts_10m": pl.Float64,
    "api_utc_offset_seconds": pl.Int32,
    "api_elevation_m": pl.Float64,
    "ingested_at_utc": pl.Datetime("us", "UTC"),
}

#: Stable ordering used before every write.
COLUMN_ORDER: Final[tuple[str, ...]] = tuple(HOURLY_SCHEMA)

#: Columns that make a row unique. Used for idempotent upserts.
DEDUP_KEYS: Final[tuple[str, ...]] = ("city_slug", "record_type", "timestamp_utc")

#: Measured columns. Missing values stay null (never fabricated) except for these
#: accumulators, where "no row-level value" genuinely means "nothing fell".
FILL_DEFAULTS: Final[dict[str, float | int]] = {
    "precipitation": 0.0,
    "rain": 0.0,
    "snowfall": 0.0,
    "weather_code": -1,
}

#: Sentinel documented in the README for an unreported WMO weather code.
UNKNOWN_WEATHER_CODE: Final[int] = -1

#: Numeric measurement columns (everything else is metadata).
MEASUREMENT_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "weather_code",
)

#: Measurement columns whose canonical dtype is integral.
INTEGER_MEASUREMENT_COLUMNS: Final[frozenset[str]] = frozenset({"weather_code"})

PARQUET_COMPRESSION: Final[str] = "snappy"
