"""Unit tests for the transform layer, focused on edge cases."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from src.config import COLUMN_ORDER, FILL_DEFAULTS, HOURLY_SCHEMA, UNKNOWN_WEATHER_CODE
from src.extract import RawPayload
from src.transform import (
    TransformError,
    clean_and_cast,
    coerce_float,
    coerce_int,
    describe_frame,
    drop_future_observations,
    empty_hourly_frame,
    enforce_schema,
    parse_utc_timestamp,
    transform_many,
    transform_payload,
)


# --------------------------------------------------------------------------------------
# Timestamp handling
# --------------------------------------------------------------------------------------


def test_parse_utc_timestamp_treats_naive_values_as_api_offset():
    moment = parse_utc_timestamp("2026-10-12T00:00", utc_offset_seconds=7200)
    assert moment == datetime(2026, 10, 11, 22, 0, tzinfo=timezone.utc)


def test_parse_utc_timestamp_converts_negative_offsets():
    moment = parse_utc_timestamp("2026-10-12T00:00", utc_offset_seconds=-18000)
    assert moment == datetime(2026, 10, 12, 5, 0, tzinfo=timezone.utc)


def test_parse_utc_timestamp_accepts_z_suffix():
    assert parse_utc_timestamp("2026-10-12T03:00:00Z") == datetime(2026, 10, 12, 3, 0, tzinfo=timezone.utc)


def test_parse_utc_timestamp_respects_explicit_offset():
    assert parse_utc_timestamp("2026-10-12T03:00:00+05:30") == datetime(
        2026, 10, 11, 21, 30, tzinfo=timezone.utc
    )


def test_parse_utc_timestamp_accepts_datetime_objects():
    assert parse_utc_timestamp(datetime(2026, 10, 12, 3, 0)) == datetime(
        2026, 10, 12, 3, 0, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("value", ["", "   ", "not-a-date", None, 42])
def test_parse_utc_timestamp_rejects_unusable_values(value):
    with pytest.raises(TransformError):
        parse_utc_timestamp(value)


# --------------------------------------------------------------------------------------
# Scalar coercion
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), (1.5, 1.5), ("2.5", 2.5), ("", None), (None, None), (True, None), ("abc", None), ([], None)],
)
def test_coerce_float(value, expected):
    assert coerce_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(3, 3), (3.7, 3), ("4", 4), ("x", None), (None, None), (True, None)]
)
def test_coerce_int(value, expected):
    assert coerce_int(value) == expected


# --------------------------------------------------------------------------------------
# Payload -> frame
# --------------------------------------------------------------------------------------


def test_transform_payload_produces_canonical_schema(make_raw_payload):
    frame = transform_payload(make_raw_payload(hours=6))
    assert dict(frame.schema) == dict(HOURLY_SCHEMA)
    assert frame.columns == list(COLUMN_ORDER)
    assert frame.height == 6


def test_transform_payload_derives_date_and_hour(make_raw_payload):
    frame = transform_payload(
        make_raw_payload(start=datetime(2026, 10, 12, 23), hours=2)
    )
    assert frame["date_utc"].to_list() == [date(2026, 10, 12), date(2026, 10, 13)]
    assert frame["hour_utc"].to_list() == [23, 0]


def test_transform_payload_records_envelope_metadata(make_raw_payload):
    frame = transform_payload(make_raw_payload(city_slug="tokyo", record_type="forecast"))
    assert frame["city"].unique().to_list() == ["Tokyo"]
    assert frame["city_slug"].unique().to_list() == ["tokyo"]
    assert frame["record_type"].unique().to_list() == ["forecast"]
    assert frame["api_elevation_m"].to_list() == [11.0] * frame.height


def test_transform_payload_handles_missing_variable(make_raw_payload):
    frame = transform_payload(make_raw_payload(omit=("relative_humidity_2m",)))
    assert frame.height == 48
    assert frame["relative_humidity_2m"].null_count() == 48
    assert frame["temperature_2m"].null_count() == 0


def test_transform_payload_handles_missing_elevation(make_raw_payload):
    frame = transform_payload(make_raw_payload(elevation=None))
    assert frame["api_elevation_m"].null_count() == frame.height


def test_transform_payload_applies_documented_fill_defaults(make_raw_payload, make_hourly_payload):
    payload = make_hourly_payload(hours=3)
    payload["hourly"]["precipitation"] = [None, None, None]
    payload["hourly"]["weather_code"] = [None, None, None]

    frame = transform_payload(make_raw_payload(payload=payload))
    assert frame["precipitation"].to_list() == [0.0, 0.0, 0.0]
    assert frame["weather_code"].to_list() == [UNKNOWN_WEATHER_CODE] * 3
    assert FILL_DEFAULTS["precipitation"] == 0.0


def test_transform_payload_leaves_measurement_nulls_untouched(make_raw_payload, make_hourly_payload):
    payload = make_hourly_payload(hours=3)
    payload["hourly"]["temperature_2m"] = [1.0, None, 3.0]
    frame = transform_payload(make_raw_payload(payload=payload))
    assert frame["temperature_2m"].null_count() == 1


def test_transform_payload_returns_empty_frame_when_hourly_is_empty(make_raw_payload, make_hourly_payload):
    frame = transform_payload(make_raw_payload(payload=make_hourly_payload(hours=0)))
    assert frame.height == 0
    assert dict(frame.schema) == dict(HOURLY_SCHEMA)


def test_transform_payload_returns_empty_frame_when_hourly_missing(make_raw_payload):
    frame = transform_payload(make_raw_payload(payload={"latitude": 1.0}))
    assert frame.height == 0


def test_transform_payload_returns_empty_frame_when_hourly_is_not_a_mapping(make_raw_payload):
    frame = transform_payload(make_raw_payload(payload={"hourly": "oops"}))
    assert frame.height == 0


def test_transform_payload_skips_unparseable_timestamps(make_raw_payload, make_hourly_payload):
    payload = make_hourly_payload(hours=3)
    payload["hourly"]["time"] = ["2026-10-12T00:00", "not-a-timestamp", "2026-10-12T02:00"]
    frame = transform_payload(make_raw_payload(payload=payload))
    assert frame.height == 2
    assert frame["hour_utc"].to_list() == [0, 2]


def test_transform_payload_tolerates_short_parallel_arrays(make_raw_payload, make_hourly_payload):
    payload = make_hourly_payload(hours=4)
    payload["hourly"]["temperature_2m"] = [1.0, 2.0]
    frame = transform_payload(make_raw_payload(payload=payload))
    assert frame.height == 4
    assert frame["temperature_2m"].null_count() == 2


def test_transform_payload_rejects_non_sequence_variable(make_raw_payload, make_hourly_payload):
    payload = make_hourly_payload(hours=2)
    payload["hourly"]["temperature_2m"] = 21.0
    frame = transform_payload(make_raw_payload(payload=payload))
    assert frame.height == 0


def test_transform_payload_converts_to_utc(make_raw_payload):
    frame = transform_payload(
        make_raw_payload(start=datetime(2026, 10, 12, 0), hours=1, utc_offset_seconds=3600)
    )
    assert frame["timestamp_utc"].to_list() == [datetime(2026, 10, 11, 23, 0, tzinfo=timezone.utc)]


# --------------------------------------------------------------------------------------
# Observation/forecast boundary
# --------------------------------------------------------------------------------------


def test_drop_future_observations_removes_modelled_hours(make_raw_payload):
    raw = make_raw_payload(
        record_type="observation",
        start=datetime(2026, 10, 12, 0),
        hours=48,
        requested_at_utc=datetime(2026, 10, 12, 5, 30, tzinfo=timezone.utc),
    )
    frame = transform_payload(raw)
    assert frame.height == 6
    assert frame["timestamp_utc"].max() <= datetime(2026, 10, 12, 5, 0, tzinfo=timezone.utc)


def test_future_hours_are_kept_for_forecasts(make_raw_payload):
    raw = make_raw_payload(
        record_type="forecast",
        start=datetime(2026, 10, 12, 0),
        hours=48,
        requested_at_utc=datetime(2026, 10, 12, 5, 30, tzinfo=timezone.utc),
    )
    assert transform_payload(raw).height == 48


def test_drop_future_observations_is_a_noop_for_empty_frames(make_raw_payload):
    raw = make_raw_payload(record_type="observation", hours=0)
    empty = empty_hourly_frame()
    assert drop_future_observations(empty, raw).height == 0


# --------------------------------------------------------------------------------------
# Schema enforcement + aggregation
# --------------------------------------------------------------------------------------


def test_enforce_schema_adds_missing_columns_and_orders_them():
    skinny = pl.DataFrame({"city_slug": ["london"], "timestamp_utc": [datetime(2026, 10, 12)]})
    repaired = enforce_schema(skinny)
    assert repaired.columns == list(COLUMN_ORDER)
    for name, dtype in HOURLY_SCHEMA.items():
        assert repaired.schema[name] == dtype, name


def test_enforce_schema_is_idempotent(canonical_frame: pl.DataFrame):
    once = enforce_schema(canonical_frame)
    twice = enforce_schema(once)
    assert once.equals(twice)


def test_enforce_schema_casts_non_strictly():
    bad = pl.DataFrame({"city_slug": ["london"], "hour_utc": ["not-an-int"]})
    repaired = enforce_schema(bad)
    assert repaired.schema["hour_utc"] == HOURLY_SCHEMA["hour_utc"]
    assert repaired["hour_utc"].to_list() == [None]


def test_clean_and_cast_deduplicates_and_sorts(make_raw_payload):
    raw = make_raw_payload(hours=3)
    frame = transform_payload(raw)
    duplicated = pl.concat([frame, frame], how="vertical")
    cleaned = clean_and_cast(duplicated)
    assert cleaned.height == frame.height
    assert cleaned["timestamp_utc"].to_list() == sorted(cleaned["timestamp_utc"].to_list())


def test_transform_many_combines_payloads(make_raw_payload):
    payloads = [
        make_raw_payload(city_slug="london", record_type="observation"),
        make_raw_payload(city_slug="tokyo", record_type="forecast"),
    ]
    frame = transform_many(payloads)
    assert frame.height == 96
    assert sorted(frame["city_slug"].unique().to_list()) == ["london", "tokyo"]


def test_transform_many_without_payloads_returns_empty_frame():
    assert transform_many([]).height == 0


def test_describe_frame_reports_expected_fields(canonical_frame: pl.DataFrame):
    summary = describe_frame(canonical_frame)
    assert summary["rows"] == canonical_frame.height
    assert summary["cities"] == 2
    assert summary["temperature_nulls"] == 0


def test_describe_frame_handles_empty_frame():
    assert describe_frame(empty_hourly_frame()) == {"rows": 0}
