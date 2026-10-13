"""Tests for the DuckDB analytics layer."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest

from src.analytics import (
    DAILY_FILENAME,
    ROLLING_FILENAME,
    SUMMARY_COLUMNS,
    AnalyticsError,
    build_summary_frame,
    connect,
    count_parquet_files,
    create_views,
    export_query,
    run_analytics,
    sql_literal,
)
from src.config import SUMMARY_FILENAME

DAY_ONE = dt.date(2026, 10, 1)
DAY_TWO = dt.date(2026, 10, 2)


def utc(*args: int) -> dt.datetime:
    return dt.datetime(*args, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------------------
# Helpers and guards
# --------------------------------------------------------------------------------------


def test_sql_literal_escapes_single_quotes():
    assert sql_literal("O'Brien") == "'O''Brien'"
    assert sql_literal(Path("C:/data/lake").as_posix()) == "'C:/data/lake'"


def test_count_parquet_files_counts_partitions(analytics_lake):
    assert count_parquet_files(analytics_lake) == 4


def test_count_parquet_files_returns_zero_for_missing_tree(tmp_path):
    assert count_parquet_files(tmp_path) == 0


def test_connect_pins_the_session_timezone():
    con = connect()
    try:
        assert con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"
    finally:
        con.close()


def test_create_views_requires_input_data(tmp_path):
    con = connect()
    try:
        with pytest.raises(AnalyticsError, match="No Parquet files"):
            create_views(con, tmp_path)
    finally:
        con.close()


def test_create_views_is_repeatable(analytics_lake):
    con = connect()
    try:
        create_views(con, analytics_lake)
        create_views(con, analytics_lake)
        assert con.execute("SELECT count(*) FROM hourly").fetchone()[0] == 96
    finally:
        con.close()


# --------------------------------------------------------------------------------------
# Export mechanics
# --------------------------------------------------------------------------------------


def test_export_query_writes_parquet(tmp_path):
    con = connect()
    try:
        target = tmp_path / "out" / "result.parquet"
        rows = export_query(con, "SELECT 1 AS value, 'a' AS label", target)
    finally:
        con.close()

    assert rows == 1
    assert pl.read_parquet(target)["value"].to_list() == [1]


def test_export_query_leaves_no_temp_files_on_failure(tmp_path):
    con = connect()
    try:
        with pytest.raises(Exception):
            export_query(con, "SELECT * FROM missing_table", tmp_path / "out.parquet")
    finally:
        con.close()

    assert list(tmp_path.iterdir()) == []


def test_analytics_exports_are_snappy_compressed(analytics_lake):
    run_analytics(analytics_lake)
    target = analytics_lake / "analytics" / SUMMARY_FILENAME

    metadata = pq.ParquetFile(target).metadata
    codecs = {metadata.row_group(0).column(i).compression for i in range(metadata.num_columns)}
    assert codecs == {"SNAPPY"}


# --------------------------------------------------------------------------------------
# Analytical correctness (temperatures equal the hour index in the fixture lake)
# --------------------------------------------------------------------------------------


def test_run_analytics_creates_all_exports(analytics_lake):
    report = run_analytics(analytics_lake)

    assert set(report.exports) == {ROLLING_FILENAME, DAILY_FILENAME, SUMMARY_FILENAME}
    assert all(path.exists() for path in report.exports.values())
    assert report.input_files == 4
    assert report.row_counts[ROLLING_FILENAME] == 96
    assert report.row_counts[DAILY_FILENAME] == 4
    assert report.row_counts[SUMMARY_FILENAME] == 4


def test_run_analytics_is_idempotent_and_leaves_no_temp_files(analytics_lake):
    first = run_analytics(analytics_lake)
    second = run_analytics(analytics_lake)

    assert first.row_counts == second.row_counts
    assert list((analytics_lake / "analytics").glob("*.tmp*")) == []


def test_rolling_average_spans_a_full_24_hours(analytics_lake):
    run_analytics(analytics_lake)
    rolling = pl.read_parquet(analytics_lake / "analytics" / ROLLING_FILENAME)
    row = rolling.filter(
        (pl.col("city_slug") == "london") & (pl.col("timestamp_utc") == utc(2026, 10, 2, 0))
    ).row(0, named=True)

    assert row["rolling_window_samples"] == 24
    assert row["temperature_rolling_24h_avg_c"] == pytest.approx(12.5)
    assert row["temperature_rolling_24h_min_c"] == 1.0
    assert row["temperature_rolling_24h_max_c"] == 24.0


def test_rolling_window_is_partial_at_the_beginning(analytics_lake):
    run_analytics(analytics_lake)
    rolling = pl.read_parquet(analytics_lake / "analytics" / ROLLING_FILENAME)
    row = rolling.filter(
        (pl.col("city_slug") == "london") & (pl.col("timestamp_utc") == utc(2026, 10, 1, 0))
    ).row(0, named=True)

    assert row["rolling_window_samples"] == 1
    assert row["temperature_rolling_24h_avg_c"] == pytest.approx(0.0)


def test_daily_metrics_min_max_and_mean(analytics_lake):
    run_analytics(analytics_lake)
    daily = pl.read_parquet(analytics_lake / "analytics" / DAILY_FILENAME)
    row = daily.filter(
        (pl.col("city_slug") == "london") & (pl.col("observation_date") == DAY_ONE)
    ).row(0, named=True)

    assert row["temperature_min_c"] == pytest.approx(0.0)
    assert row["temperature_max_c"] == pytest.approx(23.0)
    assert row["temperature_avg_c"] == pytest.approx(11.5)
    assert row["temperature_range_c"] == pytest.approx(23.0)
    assert row["hourly_samples"] == 24
    assert row["observation_samples"] == 24
    assert row["forecast_samples"] == 0


def test_daily_metrics_second_day_is_a_separate_bucket(analytics_lake):
    run_analytics(analytics_lake)
    daily = pl.read_parquet(analytics_lake / "analytics" / DAILY_FILENAME)
    row = daily.filter(
        (pl.col("city_slug") == "tokyo") & (pl.col("observation_date") == DAY_TWO)
    ).row(0, named=True)

    assert row["temperature_min_c"] == pytest.approx(24.0)
    assert row["temperature_max_c"] == pytest.approx(47.0)


def test_daily_precipitation_total(analytics_lake):
    run_analytics(analytics_lake)
    daily = pl.read_parquet(analytics_lake / "analytics" / DAILY_FILENAME)
    row = daily.filter(
        (pl.col("city_slug") == "london") & (pl.col("observation_date") == DAY_ONE)
    ).row(0, named=True)

    assert row["precipitation_total_mm"] == pytest.approx(6.0)


def test_summary_view_exposes_expected_columns_and_extremes(analytics_lake):
    summary = build_summary_frame(analytics_lake)

    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert summary.height == 4

    row = summary.filter(
        (pl.col("city_slug") == "london") & (pl.col("observation_date") == DAY_ONE)
    ).row(0, named=True)

    assert row["warmest_hour_utc"] == utc(2026, 10, 1, 23)
    assert row["coldest_hour_utc"] == utc(2026, 10, 1, 0)
    assert row["rolling_avg_c"] == pytest.approx(5.75)
    assert row["rolling_min_c"] == pytest.approx(0.0)
    assert row["rolling_max_c"] == pytest.approx(23.0)
    assert row["generated_at_utc"] is not None


def test_day_boundary_is_evaluated_in_utc(analytics_lake):
    """An hour at 23:00Z must stay on its own UTC date, never shifted by the host zone."""
    run_analytics(analytics_lake)
    daily = pl.read_parquet(analytics_lake / "analytics" / DAILY_FILENAME)

    assert set(daily["observation_date"].unique().to_list()) == {DAY_ONE, DAY_TWO}
    day_one_max = daily.filter(pl.col("observation_date") == DAY_ONE)["temperature_max_c"].max()
    assert day_one_max == pytest.approx(23.0)


def test_run_analytics_does_not_mutate_the_lake(analytics_lake):
    before = sorted(path.name for path in (analytics_lake / "hourly").rglob("*.parquet"))
    run_analytics(analytics_lake)
    after = sorted(path.name for path in (analytics_lake / "hourly").rglob("*.parquet"))

    assert before == after
