"""Unit tests for the extraction layer (retry policy, error handling, envelopes)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
import requests
import responses

from src.config import ARCHIVE_ENDPOINT, FORECAST_ENDPOINT, CITIES, get_city
from src.extract import (
    ExtractionError,
    HttpClient,
    HttpConfig,
    build_archive_params,
    build_forecast_params,
    extract_many,
    fetch_forecast,
    fetch_observation,
    recent_past_days,
    select_observation_endpoint,
)
from conftest import json_response, query_params

TODAY = date(2026, 10, 12)


@pytest.fixture()
def sleeps() -> list[float]:
    return []


@pytest.fixture()
def client(sleeps: list[float]) -> HttpClient:
    """A client with zero jitter, so the backoff schedule is exactly assertable."""
    return HttpClient(
        config=HttpConfig(
            max_attempts=3,
            backoff_base_seconds=0.5,
            backoff_max_seconds=30.0,
            jitter_ratio=0.0,
            timeout_seconds=1.0,
        ),
        sleep=sleeps.append,
    )


# --------------------------------------------------------------------------------------
# Request builders
# --------------------------------------------------------------------------------------


def test_build_forecast_params_contains_expected_keys():
    params = build_forecast_params(get_city("london"), forecast_days=3, past_days=2)
    assert params["latitude"] == pytest.approx(51.5074)
    assert params["longitude"] == pytest.approx(-0.1278)
    assert params["timezone"] == "UTC"
    assert params["forecast_days"] == 3
    assert params["past_days"] == 2
    assert "temperature_2m" in params["hourly"]
    assert "wind_gusts_10m" in params["hourly"]


@pytest.mark.parametrize("forecast_days", [0, -1])
def test_build_forecast_params_rejects_bad_forecast_days(forecast_days: int):
    with pytest.raises(ValueError):
        build_forecast_params(get_city("london"), forecast_days=forecast_days)


@pytest.mark.parametrize("past_days", [-1, 93])
def test_build_forecast_params_rejects_bad_past_days(past_days: int):
    with pytest.raises(ValueError):
        build_forecast_params(get_city("london"), past_days=past_days)


def test_build_archive_params_uses_iso_dates():
    params = build_archive_params(get_city("tokyo"), date(2026, 1, 1), date(2026, 1, 5))
    assert params["start_date"] == "2026-01-01"
    assert params["end_date"] == "2026-01-05"


def test_build_archive_params_rejects_reversed_range():
    with pytest.raises(ValueError):
        build_archive_params(get_city("tokyo"), date(2026, 1, 5), date(2026, 1, 1))


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (date(2026, 10, 1), ARCHIVE_ENDPOINT),
        (date(2026, 10, 6), ARCHIVE_ENDPOINT),
        (date(2026, 10, 7), ARCHIVE_ENDPOINT),  # exactly today - 5 days: still archived
        (date(2026, 10, 8), FORECAST_ENDPOINT),
        (date(2026, 10, 12), FORECAST_ENDPOINT),
    ],
)
def test_select_observation_endpoint_routes_by_archive_lag(target: date, expected: str):
    assert select_observation_endpoint(target, today=TODAY) == expected


def test_recent_past_days_is_clamped_to_supported_range():
    assert recent_past_days(TODAY, TODAY) == 0
    assert recent_past_days(date(2026, 10, 8), TODAY) == 4
    assert recent_past_days(date(2020, 1, 1), TODAY) == 92


# --------------------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------------------


@responses.activate
def test_get_json_returns_decoded_body(client: HttpClient):
    responses.add(responses.GET, FORECAST_ENDPOINT, json={"ok": True}, status=200)
    assert client.get_json(FORECAST_ENDPOINT) == {"ok": True}
    assert len(responses.calls) == 1


@responses.activate
def test_retries_transient_500_then_succeeds(client: HttpClient, sleeps: list[float]):
    responses.add(responses.GET, FORECAST_ENDPOINT, status=500, body="boom")
    responses.add(responses.GET, FORECAST_ENDPOINT, json={"ok": True}, status=200)

    assert client.get_json(FORECAST_ENDPOINT) == {"ok": True}
    assert len(responses.calls) == 2
    assert sleeps == [0.5]


@responses.activate
def test_exhausted_retries_raise_extraction_error(client: HttpClient, sleeps: list[float]):
    for _ in range(3):
        responses.add(responses.GET, FORECAST_ENDPOINT, status=503, body="unavailable")

    with pytest.raises(ExtractionError) as excinfo:
        client.get_json(FORECAST_ENDPOINT)
    assert "exhausted 3 attempts" in str(excinfo.value)
    assert sleeps == [0.5, 1.0]


@responses.activate
def test_non_retryable_4xx_fails_fast(client: HttpClient, sleeps: list[float]):
    responses.add(responses.GET, FORECAST_ENDPOINT, status=400, body="bad request")

    with pytest.raises(ExtractionError, match="non-retryable"):
        client.get_json(FORECAST_ENDPOINT)
    assert len(responses.calls) == 1
    assert sleeps == []


@responses.activate
def test_retry_after_header_is_honoured(client: HttpClient, sleeps: list[float]):
    responses.add(responses.GET, FORECAST_ENDPOINT, status=429, headers={"Retry-After": "2"})
    responses.add(responses.GET, FORECAST_ENDPOINT, json={"ok": True}, status=200)

    assert client.get_json(FORECAST_ENDPOINT) == {"ok": True}
    assert sleeps == [2.0]


@responses.activate
def test_retries_on_connection_error(client: HttpClient, sleeps: list[float]):
    responses.add(responses.GET, FORECAST_ENDPOINT, body=requests.ConnectionError("refused"))
    responses.add(responses.GET, FORECAST_ENDPOINT, json={"ok": True}, status=200)

    assert client.get_json(FORECAST_ENDPOINT) == {"ok": True}
    assert sleeps == [0.5]


@responses.activate
def test_invalid_json_body_raises(client: HttpClient):
    responses.add(
        responses.GET, FORECAST_ENDPOINT, body="<html>nope</html>", status=200, content_type="text/html"
    )
    with pytest.raises(ExtractionError, match="not valid JSON"):
        client.get_json(FORECAST_ENDPOINT)


def test_backoff_is_exponential_and_capped():
    config = HttpConfig(
        max_attempts=8, backoff_base_seconds=1.0, backoff_max_seconds=4.0, jitter_ratio=0.0
    )
    client = HttpClient(config=config)
    assert [client._compute_backoff(attempt) for attempt in range(1, 6)] == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_jitter_stays_within_configured_ratio():
    config = HttpConfig(backoff_base_seconds=4.0, jitter_ratio=0.25)
    client = HttpClient(config=config)
    for _ in range(50):
        delay = client._compute_backoff(1)
        assert 4.0 <= delay <= 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"backoff_base_seconds": 0},
        {"backoff_base_seconds": 5.0, "backoff_max_seconds": 1.0},
        {"jitter_ratio": 1.5},
    ],
)
def test_http_config_validates_inputs(kwargs: dict[str, float]):
    with pytest.raises(ValueError):
        HttpConfig(**kwargs)


# --------------------------------------------------------------------------------------
# Payload fetching
# --------------------------------------------------------------------------------------


@responses.activate
def test_fetch_forecast_builds_envelope(client: HttpClient, make_hourly_payload):
    responses.add(responses.GET, FORECAST_ENDPOINT, json=make_hourly_payload(), status=200)

    raw = fetch_forecast(client, get_city("london"))
    assert raw.record_type == "forecast"
    assert raw.label == "london/forecast"
    assert raw.endpoint == FORECAST_ENDPOINT
    params = query_params(responses.calls[0].request)
    assert params["timezone"] == "UTC"
    assert "temperature_2m" in params["hourly"]


@responses.activate
def test_fetch_observation_uses_archive_for_old_dates(client: HttpClient, make_hourly_payload):
    responses.add(responses.GET, ARCHIVE_ENDPOINT, json=make_hourly_payload(), status=200)

    raw = fetch_observation(client, get_city("tokyo"), date(2026, 1, 1), today=TODAY)
    assert raw.endpoint == ARCHIVE_ENDPOINT
    params = query_params(responses.calls[0].request)
    assert params["start_date"] == "2026-01-01"
    assert params["end_date"] == "2026-01-01"


@responses.activate
def test_fetch_observation_uses_recent_window_for_new_dates(client: HttpClient, make_hourly_payload):
    responses.add(responses.GET, FORECAST_ENDPOINT, json=make_hourly_payload(), status=200)

    raw = fetch_observation(client, get_city("london"), date(2026, 10, 10), today=TODAY)
    assert raw.endpoint == FORECAST_ENDPOINT
    params = query_params(responses.calls[0].request)
    assert params["past_days"] == "2"
    assert params["forecast_days"] == "1"


@responses.activate
def test_missing_hourly_block_raises(client: HttpClient):
    responses.add(responses.GET, FORECAST_ENDPOINT, json={"latitude": 1.0}, status=200)
    with pytest.raises(ExtractionError, match="hourly"):
        fetch_forecast(client, get_city("london"))


@responses.activate
def test_api_reported_error_raises(client: HttpClient):
    responses.add(
        responses.GET,
        FORECAST_ENDPOINT,
        json={"error": True, "reason": "Latitude must be in range"},
        status=200,
    )
    with pytest.raises(ExtractionError, match="Latitude must be in range"):
        fetch_forecast(client, get_city("london"))


@responses.activate
def test_extract_many_isolates_partial_failures(client: HttpClient, make_hourly_payload):
    payload = make_hourly_payload()

    def callback(request: requests.PreparedRequest):
        params = query_params(request)
        is_tokyo = float(params["latitude"]) > 30 and float(params["longitude"]) > 100
        if is_tokyo:
            return 400, {"Content-Type": "text/plain"}, "bad request"
        return json_response(payload)

    responses.add_callback(responses.GET, FORECAST_ENDPOINT, callback=callback)
    responses.add_callback(responses.GET, ARCHIVE_ENDPOINT, callback=callback)

    report = extract_many(client, CITIES, [date(2026, 1, 1)], today=TODAY)
    assert report.succeeded == 4
    assert report.failed == 2
    assert {target.split("/")[0] for target, _ in report.errors} == {"tokyo"}


@responses.activate
def test_extract_many_fetches_forecast_once_per_city(client: HttpClient, make_hourly_payload):
    responses.add_callback(
        responses.GET,
        FORECAST_ENDPOINT,
        callback=lambda request: json_response(make_hourly_payload()),
    )
    responses.add_callback(
        responses.GET,
        ARCHIVE_ENDPOINT,
        callback=lambda request: json_response(make_hourly_payload()),
    )

    report = extract_many(client, CITIES, [date(2026, 1, 1), date(2026, 1, 2)], today=TODAY)
    forecasts = [raw for raw in report.payloads if raw.record_type == "forecast"]
    observations = [raw for raw in report.payloads if raw.record_type == "observation"]
    assert len(forecasts) == len(CITIES)
    assert len(observations) == len(CITIES) * 2


@responses.activate
def test_fetch_forecast_propagates_extraction_errors():
    responses.add(responses.GET, FORECAST_ENDPOINT, status=500, body="boom")
    client = HttpClient(
        config=HttpConfig(max_attempts=1, jitter_ratio=0.0), sleep=lambda _seconds: None
    )
    with pytest.raises(ExtractionError, match="exhausted 1 attempts"):
        fetch_forecast(client, get_city("london"))
