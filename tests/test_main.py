"""Tests for the CLI entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from datetime import date
from pathlib import Path

import pytest
import responses

from src.config import ARCHIVE_ENDPOINT, FORECAST_ENDPOINT, ConfigurationError
from src.main import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    RunReport,
    build_parser,
    main,
    resolve_city_slugs,
    resolve_target_dates,
    write_report,
)
from conftest import json_response

TODAY = date(2026, 10, 12)


def namespace(**overrides) -> argparse.Namespace:
    values = {"dates": None, "backfill": None}
    values.update(overrides)
    return argparse.Namespace(**values)


# --------------------------------------------------------------------------------------
# Argument parsing and date resolution
# --------------------------------------------------------------------------------------


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.backfill is None
    assert args.dates is None
    assert args.cities is None
    assert args.skip_analytics is False
    assert args.max_attempts == 5
    assert args.log_level == "INFO"


def test_parser_accepts_repeated_dates_and_backfill():
    args = build_parser().parse_args(["--date", "2026-10-01", "--date", "2026-10-02"])
    assert args.dates == ["2026-10-01", "2026-10-02"]
    assert build_parser().parse_args(["--backfill", "3"]).backfill == 3


def test_resolve_target_dates_defaults_to_today():
    assert resolve_target_dates(namespace(), today=TODAY) == [TODAY]


def test_resolve_target_dates_backfill_is_oldest_first():
    dates = resolve_target_dates(namespace(backfill=3), today=TODAY)
    assert dates == [date(2026, 10, 10), date(2026, 10, 11), date(2026, 10, 12)]


def test_resolve_target_dates_backfill_of_one():
    assert resolve_target_dates(namespace(backfill=1), today=TODAY) == [TODAY]


def test_resolve_target_dates_sorts_and_deduplicates_explicit_dates():
    resolved = resolve_target_dates(namespace(dates=["2026-10-02", "2026-10-01", "2026-10-02"]), today=TODAY)
    assert resolved == [date(2026, 10, 1), date(2026, 10, 2)]


def test_resolve_target_dates_rejects_conflicting_flags():
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        resolve_target_dates(namespace(dates=["2026-10-01"], backfill=2), today=TODAY)


def test_resolve_target_dates_rejects_future_dates():
    with pytest.raises(ConfigurationError, match="future date"):
        resolve_target_dates(namespace(dates=["2026-10-13"]), today=TODAY)


def test_resolve_target_dates_rejects_malformed_date():
    with pytest.raises(ConfigurationError, match="invalid --date"):
        resolve_target_dates(namespace(dates=["12/10/2026"]), today=TODAY)


@pytest.mark.parametrize("backfill", [0, -3, 1000])
def test_resolve_target_dates_rejects_out_of_range_backfill(backfill: int):
    with pytest.raises(ConfigurationError, match="--backfill must be between"):
        resolve_target_dates(namespace(backfill=backfill), today=TODAY)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("london", ("london",)), (" london , tokyo ", ("london", "tokyo"))],
)
def test_resolve_city_slugs(raw, expected):
    assert resolve_city_slugs(raw) == expected


def test_resolve_city_slugs_rejects_empty_list():
    with pytest.raises(ConfigurationError):
        resolve_city_slugs(" , ")


def test_resolve_city_slugs_rejects_unknown_city():
    with pytest.raises(ConfigurationError, match="Unknown city"):
        resolve_city_slugs("atlantis")


def test_resolve_city_slugs_canonicalises_display_names():
    assert resolve_city_slugs("New York") == ("new_york",)


# --------------------------------------------------------------------------------------
# Run reports
# --------------------------------------------------------------------------------------


def test_write_report_persists_json(tmp_path):
    report = RunReport(started_at_utc="2026-10-12T00:00:00+00:00", data_dir=tmp_path.as_posix())
    target = write_report(tmp_path / "nested" / "report.json", report)

    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    assert payload["data_dir"] == tmp_path.as_posix()
    assert payload["version"]


# --------------------------------------------------------------------------------------
# End-to-end CLI runs (HTTP mocked at the transport layer with responses)
# --------------------------------------------------------------------------------------


def _today_payload(make_hourly_payload):
    now = dt.datetime.now(tz=dt.timezone.utc)
    start = dt.datetime.combine(now.date() - dt.timedelta(days=1), dt.time(0, 0))
    return make_hourly_payload(start=start, hours=72)


@responses.activate
def test_main_runs_the_full_pipeline(tmp_path, make_hourly_payload):
    payload = _today_payload(make_hourly_payload)
    responses.add_callback(responses.GET, FORECAST_ENDPOINT, callback=lambda request: json_response(payload))
    responses.add_callback(responses.GET, ARCHIVE_ENDPOINT, callback=lambda request: json_response(payload))

    data_dir = tmp_path / "data"
    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--cities",
            "london,tokyo",
            "--max-attempts",
            "1",
            "--report-json",
            str(report_path),
        ]
    )

    assert exit_code == EXIT_OK
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["loaded_rows"] > 0
    assert report["partitions_touched"] > 0
    assert sorted(report["cities"]) == ["london", "tokyo"]
    assert report["extraction_errors"] == []
    assert (data_dir / "analytics" / "summary.parquet").exists()
    assert report["analytics"]["row_counts"]["summary.parquet"] > 0
    assert list(data_dir.rglob("*.tmp*")) == []


@responses.activate
def test_main_can_skip_analytics(tmp_path, make_hourly_payload):
    payload = _today_payload(make_hourly_payload)
    responses.add_callback(responses.GET, FORECAST_ENDPOINT, callback=lambda request: json_response(payload))
    responses.add_callback(responses.GET, ARCHIVE_ENDPOINT, callback=lambda request: json_response(payload))

    data_dir = tmp_path / "data"
    exit_code = main(
        ["--data-dir", str(data_dir), "--cities", "london", "--max-attempts", "1", "--skip-analytics"]
    )

    assert exit_code == EXIT_OK
    assert list((data_dir / "hourly").rglob("*.parquet"))
    assert not (data_dir / "analytics").exists()


@responses.activate
def test_main_can_skip_the_forecast(tmp_path, make_hourly_payload):
    payload = _today_payload(make_hourly_payload)
    responses.add_callback(responses.GET, FORECAST_ENDPOINT, callback=lambda request: json_response(payload))

    data_dir = tmp_path / "data"
    exit_code = main(
        ["--data-dir", str(data_dir), "--cities", "london", "--max-attempts", "1", "--skip-forecast"]
    )

    assert exit_code == EXIT_OK
    assert list((data_dir / "hourly").rglob("*__forecast.parquet")) == []
    assert list((data_dir / "hourly").rglob("*__observation.parquet"))


@responses.activate
def test_main_fails_when_every_request_errors(tmp_path):
    responses.add_callback(responses.GET, FORECAST_ENDPOINT, callback=lambda request: (400, {}, "nope"))
    responses.add_callback(responses.GET, ARCHIVE_ENDPOINT, callback=lambda request: (400, {}, "nope"))

    exit_code = main(["--data-dir", str(tmp_path / "data"), "--cities", "london", "--max-attempts", "1"])
    assert exit_code == EXIT_FAILURE


def test_main_returns_usage_error_for_malformed_date(tmp_path):
    assert main(["--data-dir", str(tmp_path), "--date", "12-10-2026"]) == EXIT_USAGE


def test_main_returns_usage_error_for_conflicting_flags(tmp_path):
    assert main(["--data-dir", str(tmp_path), "--date", "2026-10-01", "--backfill", "2"]) == EXIT_USAGE


def test_main_returns_usage_error_when_nothing_would_be_fetched(tmp_path):
    code = main(
        ["--data-dir", str(tmp_path), "--skip-forecast", "--skip-observations"]
    )
    assert code == EXIT_USAGE


@responses.activate
def test_main_is_idempotent_across_runs(tmp_path, make_hourly_payload):
    payload = _today_payload(make_hourly_payload)
    responses.add_callback(responses.GET, FORECAST_ENDPOINT, callback=lambda request: json_response(payload))
    responses.add_callback(responses.GET, ARCHIVE_ENDPOINT, callback=lambda request: json_response(payload))

    data_dir = tmp_path / "data"
    argv = ["--data-dir", str(data_dir), "--cities", "london", "--max-attempts", "1"]

    assert main(argv) == EXIT_OK
    first_rows = sum(1 for _ in data_dir.rglob("*.parquet"))

    assert main(argv) == EXIT_OK
    second_rows = sum(1 for _ in data_dir.rglob("*.parquet"))

    assert first_rows == second_rows
