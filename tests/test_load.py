"""Tests for the load layer: partitioning, atomic writes and idempotent upserts."""

from __future__ import annotations

import datetime as dt
from datetime import date

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest

from src.config import DEDUP_KEYS
from src.load import (
    LoadError,
    atomic_write_parquet,
    hourly_glob,
    load_hourly,
    partition_dir,
    partition_path,
    read_partition,
    upsert_partition,
    verify_partition,
)
from src.transform import empty_hourly_frame


# --------------------------------------------------------------------------------------
# Partition layout
# --------------------------------------------------------------------------------------


def test_partition_dir_follows_hive_layout(tmp_path):
    path = partition_dir(tmp_path, date(2026, 10, 5))
    assert path.as_posix().endswith("hourly/year=2026/month=10/day=05")


def test_partition_dir_zero_pads_month_and_day(tmp_path):
    assert partition_dir(tmp_path, date(2026, 1, 1)).as_posix().endswith("year=2026/month=01/day=01")


def test_partition_path_is_deterministic(tmp_path):
    first = partition_path(tmp_path, "london", "forecast", date(2026, 10, 5))
    second = partition_path(tmp_path, "london", "forecast", date(2026, 10, 5))
    assert first == second
    assert first.name == "london__forecast.parquet"


@pytest.mark.parametrize(("slug", "record_type"), [("", "forecast"), ("london", "")])
def test_partition_path_rejects_empty_parts(tmp_path, slug, record_type):
    with pytest.raises(LoadError):
        partition_path(tmp_path, slug, record_type, date(2026, 10, 5))


def test_hourly_glob_targets_parquet_under_the_dataset(tmp_path):
    glob = hourly_glob(tmp_path)
    assert glob.endswith("hourly/**/*.parquet")
    assert "\\" not in glob


# --------------------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------------------


def test_atomic_write_creates_file_without_temp_leftovers(tmp_path, canonical_frame):
    target = tmp_path / "nested" / "part.parquet"
    atomic_write_parquet(canonical_frame, target)

    assert target.exists()
    assert list(target.parent.glob("*.tmp*")) == []
    assert read_partition(target).height == canonical_frame.height


def test_atomic_write_replaces_existing_file(tmp_path, canonical_frame):
    target = tmp_path / "part.parquet"
    atomic_write_parquet(canonical_frame.head(2), target)
    atomic_write_parquet(canonical_frame, target)

    assert read_partition(target).height == canonical_frame.height
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_write_removes_temp_file_and_target_on_failure(tmp_path):
    target = tmp_path / "part.parquet"
    with pytest.raises(Exception):
        atomic_write_parquet({"not": "a dataframe"}, target)  # type: ignore[arg-type]

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_preserves_previous_file_when_new_write_fails(tmp_path, canonical_frame):
    target = tmp_path / "part.parquet"
    atomic_write_parquet(canonical_frame, target)
    with pytest.raises(Exception):
        atomic_write_parquet({"not": "a dataframe"}, target)  # type: ignore[arg-type]

    assert read_partition(target).height == canonical_frame.height


def test_output_is_snappy_compressed(tmp_path, canonical_frame):
    target = tmp_path / "part.parquet"
    atomic_write_parquet(canonical_frame, target)

    metadata = pq.ParquetFile(target).metadata
    codecs = {metadata.row_group(0).column(i).compression for i in range(metadata.num_columns)}
    assert codecs == {"SNAPPY"}


def test_read_partition_returns_empty_frame_when_missing(tmp_path):
    assert read_partition(tmp_path / "missing.parquet").height == 0


def test_verify_partition_rejects_missing_file(tmp_path):
    with pytest.raises(LoadError, match="was not created"):
        verify_partition(tmp_path / "missing.parquet")


def test_verify_partition_rejects_row_count_mismatch(tmp_path, canonical_frame):
    target = tmp_path / "part.parquet"
    atomic_write_parquet(canonical_frame, target)
    with pytest.raises(LoadError, match="expected 999"):
        verify_partition(target, expected_rows=999)


# --------------------------------------------------------------------------------------
# Upsert semantics
# --------------------------------------------------------------------------------------


def test_upsert_partition_is_idempotent(tmp_path, canonical_frame):
    target = partition_path(tmp_path, "london", "observation", date(2026, 10, 12))

    first = upsert_partition(canonical_frame, target)
    second = upsert_partition(canonical_frame, target)

    assert first.rows_written == second.rows_written
    assert second.rows_dropped == second.rows_in_batch
    assert read_partition(target).height == first.rows_written
    assert target.exists()


def test_upsert_partition_merges_overlapping_batches(tmp_path, make_raw_payload):
    from src.transform import transform_many

    early = transform_many([make_raw_payload(start=dt.datetime(2026, 10, 12, 0), hours=24)])
    late = transform_many([make_raw_payload(start=dt.datetime(2026, 10, 12, 12), hours=24)])

    target = partition_path(tmp_path, "london", "observation", date(2026, 10, 12))
    upsert_partition(early, target)
    result = upsert_partition(late, target)

    frame = read_partition(target)
    assert frame.height == 36  # 24 + 24 - 12 overlapping hours
    assert result.rows_dropped == 12
    assert frame.select(list(DEDUP_KEYS)).is_duplicated().sum() == 0


# --------------------------------------------------------------------------------------
# Whole-frame loads
# --------------------------------------------------------------------------------------


def test_load_hourly_writes_one_file_per_city_type_and_day(tmp_path, canonical_frame):
    report = load_hourly(canonical_frame, tmp_path)

    assert report.files_touched == 8
    assert report.rows_written == canonical_frame.height
    assert sorted({path.name for path in report.files}) == [
        "london__forecast.parquet",
        "london__observation.parquet",
        "tokyo__forecast.parquet",
        "tokyo__observation.parquet",
    ]
    assert {path.parent.name for path in report.files} == {"day=12", "day=13"}


def test_load_hourly_is_idempotent(tmp_path, canonical_frame):
    first = load_hourly(canonical_frame, tmp_path)
    second = load_hourly(canonical_frame, tmp_path)

    assert first.rows_written == second.rows_written
    assert {part.rows_written for part in second.partitions} == {
        part.rows_written for part in first.partitions
    }


def test_load_hourly_with_empty_frame_writes_nothing(tmp_path):
    report = load_hourly(empty_hourly_frame(), tmp_path)
    assert report.files_touched == 0
    assert report.rows_written == 0
    assert list(tmp_path.rglob("*.parquet")) == []


def test_load_report_serialises_to_dict(tmp_path, canonical_frame):
    report = load_hourly(canonical_frame, tmp_path)
    payload = report.as_dict()

    assert payload["rows_written"] == report.rows_written
    assert payload["files_touched"] == report.files_touched
    assert len(payload["partitions"]) == report.files_touched


@pytest.mark.integration
def test_partitions_are_readable_with_hive_partitioning(tmp_path, canonical_frame):
    load_hourly(canonical_frame, tmp_path)

    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        rows = con.execute(
            "SELECT year, month, day, count(*) FROM read_parquet("
            + sql_literal_for_test(hourly_glob(tmp_path))
            + ", hive_partitioning = true) GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
        ).fetchall()
    finally:
        con.close()

    assert rows == [(2026, 10, 12, 96), (2026, 10, 13, 96)]


def sql_literal_for_test(value: str) -> str:
    """Tiny local helper so the test does not depend on the analytics module."""
    return "'" + value.replace("'", "''") + "'"
