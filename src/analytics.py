"""Analytics layer: DuckDB SQL over the Parquet lake.

The stage is deliberately declarative. DuckDB reads the Hive-partitioned lake
directly (no data loading step), builds three views, and exports them:

```
rolling_24h    -- hourly grain, trailing 24-hour rolling statistics per city/record type
daily_metrics  -- daily min/max/mean temperature, precipitation and wind aggregates
summary.parquet-- the joined, analysis-ready view (one row per city/record type/day)
```

Exports are written to a temporary file and moved into place with `os.replace`, the
same atomicity contract the load layer uses, so a failing query can never truncate a
previously good artefact.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import duckdb
import polars as pl

from src.config import (
    DATASET_ANALYTICS,
    DATASET_HOURLY,
    PARQUET_COMPRESSION,
    SUMMARY_FILENAME,
)

LOGGER = logging.getLogger(__name__)

ROLLING_FILENAME = "rolling_24h.parquet"
DAILY_FILENAME = "daily_metrics.parquet"

#: Trailing window length in seconds. 86399 (rather than 86400) keeps exactly the
#: current hour plus the preceding 23 hourly samples in the window.
ROLLING_WINDOW_SECONDS = 86399

ROLLING_COLUMNS: tuple[str, ...] = (
    "city_slug",
    "city",
    "record_type",
    "timestamp_utc",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "temperature_rolling_24h_avg_c",
    "temperature_rolling_24h_min_c",
    "temperature_rolling_24h_max_c",
    "precipitation_rolling_24h_mm",
    "rolling_window_samples",
)

DAILY_COLUMNS: tuple[str, ...] = (
    "city_slug",
    "city",
    "record_type",
    "observation_date",
    "temperature_min_c",
    "temperature_max_c",
    "temperature_avg_c",
    "temperature_range_c",
    "humidity_avg_pct",
    "precipitation_total_mm",
    "wind_gust_max_kmh",
    "wind_speed_avg_kmh",
    "pressure_msl_avg_hpa",
    "hourly_samples",
    "forecast_samples",
    "observation_samples",
)

SUMMARY_COLUMNS: tuple[str, ...] = DAILY_COLUMNS + (
    "rolling_avg_c",
    "rolling_min_c",
    "rolling_max_c",
    "warmest_hour_utc",
    "coldest_hour_utc",
    "generated_at_utc",
)


class AnalyticsError(RuntimeError):
    """Raised when the analytics stage cannot run (for example: no input data)."""


# --------------------------------------------------------------------------------------
# SQL helpers
# --------------------------------------------------------------------------------------


def sql_literal(value: str) -> str:
    """Return `value` as a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection pinned to UTC.

    Pinning the session time zone matters: `CAST(timestamp_utc AS DATE)` is evaluated
    in the session zone, so a local-time connection would silently shift the day
    boundary of every aggregate.
    """
    connection = duckdb.connect(database)
    connection.execute("SET TimeZone='UTC'")
    return connection


def count_parquet_files(data_dir: Path | str, dataset: str = DATASET_HOURLY) -> int:
    """Count the Parquet files available for a dataset."""
    root = Path(data_dir) / dataset
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.parquet") if path.is_file())


def _glob(data_dir: Path | str, dataset: str = DATASET_HOURLY) -> str:
    return (Path(data_dir) / dataset / "**" / "*.parquet").as_posix()


# --------------------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------------------

BASE_VIEW_SQL = """
CREATE OR REPLACE VIEW hourly AS
SELECT * FROM read_parquet({glob}, hive_partitioning = true, union_by_name = true)
"""

ROLLING_VIEW_SQL = f"""
CREATE OR REPLACE VIEW rolling_24h AS
SELECT
    city_slug,
    city,
    record_type,
    timestamp_utc,
    temperature_2m,
    apparent_temperature,
    relative_humidity_2m,
    precipitation,
    wind_speed_10m,
    avg(temperature_2m) OVER w  AS temperature_rolling_24h_avg_c,
    min(temperature_2m) OVER w  AS temperature_rolling_24h_min_c,
    max(temperature_2m) OVER w  AS temperature_rolling_24h_max_c,
    sum(precipitation) OVER w   AS precipitation_rolling_24h_mm,
    count(temperature_2m) OVER w AS rolling_window_samples
FROM hourly
WINDOW w AS (
    PARTITION BY city_slug, record_type
    ORDER BY epoch(timestamp_utc)
    RANGE BETWEEN {ROLLING_WINDOW_SECONDS} PRECEDING AND CURRENT ROW
)
"""

DAILY_VIEW_SQL = """
CREATE OR REPLACE VIEW daily_metrics AS
SELECT
    city_slug,
    city,
    record_type,
    CAST(timestamp_utc AS DATE) AS observation_date,
    min(temperature_2m) AS temperature_min_c,
    max(temperature_2m) AS temperature_max_c,
    avg(temperature_2m) AS temperature_avg_c,
    max(temperature_2m) - min(temperature_2m) AS temperature_range_c,
    avg(relative_humidity_2m) AS humidity_avg_pct,
    sum(precipitation) AS precipitation_total_mm,
    max(wind_gusts_10m) AS wind_gust_max_kmh,
    avg(wind_speed_10m) AS wind_speed_avg_kmh,
    avg(pressure_msl) AS pressure_msl_avg_hpa,
    count(*) AS hourly_samples,
    count(*) FILTER (WHERE record_type = 'forecast') AS forecast_samples,
    count(*) FILTER (WHERE record_type = 'observation') AS observation_samples
FROM hourly
GROUP BY city_slug, city, record_type, CAST(timestamp_utc AS DATE)
"""

SUMMARY_VIEW_SQL = """
CREATE OR REPLACE VIEW summary_view AS
SELECT
    d.city_slug,
    d.city,
    d.record_type,
    d.observation_date,
    d.temperature_min_c,
    d.temperature_max_c,
    d.temperature_avg_c,
    d.temperature_range_c,
    d.humidity_avg_pct,
    d.precipitation_total_mm,
    d.wind_gust_max_kmh,
    d.wind_speed_avg_kmh,
    d.pressure_msl_avg_hpa,
    d.hourly_samples,
    d.forecast_samples,
    d.observation_samples,
    r.rolling_avg_c,
    r.rolling_min_c,
    r.rolling_max_c,
    r.warmest_hour_utc,
    r.coldest_hour_utc,
    {generated_at} AS generated_at_utc
FROM daily_metrics d
LEFT JOIN (
    SELECT
        city_slug,
        record_type,
        CAST(timestamp_utc AS DATE) AS observation_date,
        avg(temperature_rolling_24h_avg_c) AS rolling_avg_c,
        min(temperature_rolling_24h_min_c) AS rolling_min_c,
        max(temperature_rolling_24h_max_c) AS rolling_max_c,
        arg_max(timestamp_utc, temperature_2m) AS warmest_hour_utc,
        arg_min(timestamp_utc, temperature_2m) AS coldest_hour_utc
    FROM rolling_24h
    GROUP BY city_slug, record_type, CAST(timestamp_utc AS DATE)
) r
  ON r.city_slug = d.city_slug
 AND r.record_type = d.record_type
 AND r.observation_date = d.observation_date
"""


@dataclass(slots=True)
class AnalyticsReport:
    """Outcome of :func:`run_analytics`."""

    input_files: int = 0
    exports: dict[str, Path] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_files": self.input_files,
            "exports": {name: path.as_posix() for name, path in self.exports.items()},
            "row_counts": dict(self.row_counts),
        }


def create_views(con: duckdb.DuckDBPyConnection, data_dir: Path | str) -> int:
    """Create the base, rolling, daily and summary views. Returns the input file count."""
    files = count_parquet_files(data_dir)
    if files == 0:
        raise AnalyticsError(f"No Parquet files found under {Path(data_dir).as_posix()}")
    con.execute(BASE_VIEW_SQL.format(glob=sql_literal(_glob(data_dir))))
    con.execute(ROLLING_VIEW_SQL)
    con.execute(DAILY_VIEW_SQL)
    con.execute(
        SUMMARY_VIEW_SQL.format(
            generated_at="CAST(" + sql_literal(datetime.now(tz=timezone.utc).isoformat()) + " AS TIMESTAMPTZ)"
        )
    )
    LOGGER.info("Analytics views created over %d Parquet file(s)", files)
    return files


def export_query(
    con: duckdb.DuckDBPyConnection,
    select_sql: str,
    target: Path | str,
    compression: str = PARQUET_COMPRESSION,
) -> int:
    """Run `select_sql` and atomically export the result to Parquet.

    Returns the number of rows written.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(f"{target_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_sql = tmp.as_posix()
        con.execute(
            f"COPY ({select_sql}) TO {sql_literal(tmp_sql)} "
            f"(FORMAT PARQUET, COMPRESSION {compression.upper()})"
        )
        os.replace(tmp, target_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    rows = int(con.execute(f"SELECT count(*) FROM ({select_sql})").fetchone()[0])
    LOGGER.info("Exported %d row(s) to %s", rows, target_path.as_posix())
    return rows


def export_view(
    con: duckdb.DuckDBPyConnection,
    view: str,
    columns: Sequence[str],
    target: Path | str,
    order_by: Sequence[str] = (),
) -> int:
    """Export a named view (a stable column subset) to a Parquet file."""
    projection = ", ".join(columns)
    select_sql = f"SELECT {projection} FROM {view}"
    if order_by:
        select_sql += " ORDER BY " + ", ".join(order_by)
    return export_query(con, select_sql, target)


def run_analytics(data_dir: Path | str, connection: duckdb.DuckDBPyConnection | None = None) -> AnalyticsReport:
    """Build the analytical views and export rolling, daily and summary artefacts."""
    con = connection or connect()
    owns_connection = connection is None
    try:
        report = AnalyticsReport()
        report.input_files = create_views(con, data_dir)
        out_dir = Path(data_dir) / DATASET_ANALYTICS

        rolling_target = out_dir / ROLLING_FILENAME
        report.row_counts[ROLLING_FILENAME] = export_view(
            con,
            "rolling_24h",
            ROLLING_COLUMNS,
            rolling_target,
            order_by=("city_slug", "record_type", "timestamp_utc"),
        )
        report.exports[ROLLING_FILENAME] = rolling_target

        daily_target = out_dir / DAILY_FILENAME
        report.row_counts[DAILY_FILENAME] = export_view(
            con,
            "daily_metrics",
            DAILY_COLUMNS,
            daily_target,
            order_by=("city_slug", "record_type", "observation_date"),
        )
        report.exports[DAILY_FILENAME] = daily_target

        summary_target = out_dir / SUMMARY_FILENAME
        report.row_counts[SUMMARY_FILENAME] = export_view(
            con,
            "summary_view",
            SUMMARY_COLUMNS,
            summary_target,
            order_by=("city_slug", "record_type", "observation_date"),
        )
        report.exports[SUMMARY_FILENAME] = summary_target

        LOGGER.info("Analytics finished: %s", report.row_counts)
        return report
    finally:
        if owns_connection:
            con.close()


def build_summary_frame(data_dir: Path | str) -> pl.DataFrame:
    """Return the analytical summary as a polars frame (convenience for CLI and tests)."""
    con = connect()
    try:
        create_views(con, data_dir)
        return con.execute(
            "SELECT " + ", ".join(SUMMARY_COLUMNS) + " FROM summary_view"
        ).pl()
    finally:
        con.close()
