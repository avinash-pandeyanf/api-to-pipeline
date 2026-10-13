"""Shared pytest fixtures for the Open-Meteo -> Parquet pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CITIES, City, get_city  # noqa: E402
from src.extract import RawPayload  # noqa: E402
from src.transform import transform_many  # noqa: E402

#: Deterministic default series for every hourly variable the pipeline knows about.
DEFAULT_SERIES: dict[str, Callable[[int], float | int]] = {
    "temperature_2m": lambda i: 10.0 + i * 0.5,
    "relative_humidity_2m": lambda i: 60.0,
    "apparent_temperature": lambda i: 9.0,
    "precipitation": lambda i: 0.0 if i % 6 else 1.5,
    "rain": lambda i: 0.0,
    "snowfall": lambda i: 0.0,
    "weather_code": lambda i: 3,
    "cloud_cover": lambda i: 25.0,
    "pressure_msl": lambda i: 1013.0,
    "surface_pressure": lambda i: 1002.0,
    "wind_speed_10m": lambda i: 11.0,
    "wind_direction_10m": lambda i: 180.0,
    "wind_gusts_10m": lambda i: 22.0,
}

DEFAULT_START = datetime(2026, 10, 12, 0, 0)


@pytest.fixture()
def london() -> City:
    return get_city("london")


@pytest.fixture()
def all_cities() -> tuple[City, ...]:
    return CITIES


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test tries to open a real socket.

    HTTP is always mocked (usually with `responses`), so a real connection attempt
    means the test would silently depend on the public internet.
    """
    import socket

    def guard(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Tests must not touch the real network; mock HTTP with responses")

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture()
def no_sleep() -> Callable[[float], None]:
    """A sleeper that does nothing, keeping retry tests instant."""

    def _sleep(_seconds: float) -> None:
        return None

    return _sleep


@pytest.fixture()
def make_hourly_payload() -> Callable[..., dict[str, Any]]:
    """Factory for Open-Meteo-shaped JSON payloads."""

    def _make(
        *,
        start: datetime = DEFAULT_START,
        hours: int = 48,
        utc_offset_seconds: int = 0,
        elevation: float | None = 11.0,
        time_format: str = "%Y-%m-%dT%H:%M",
        omit: Sequence[str] = (),
        series_overrides: Mapping[str, Callable[[int], Any]] | None = None,
    ) -> dict[str, Any]:
        times = [start + timedelta(hours=index) for index in range(hours)]
        series = dict(DEFAULT_SERIES)
        series.update(series_overrides or {})
        hourly: dict[str, Any] = {"time": [moment.strftime(time_format) for moment in times]}
        for name, generator in series.items():
            if name in omit:
                continue
            hourly[name] = [generator(index) for index in range(hours)]
        payload: dict[str, Any] = {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "generationtime_ms": 0.25,
            "utc_offset_seconds": utc_offset_seconds,
            "timezone": "UTC",
            "timezone_abbreviation": "GMT",
            "hourly_units": {"temperature_2m": "C"},
            "hourly": hourly,
        }
        if elevation is not None:
            payload["elevation"] = elevation
        return payload

    return _make


@pytest.fixture()
def make_raw_payload(make_hourly_payload: Callable[..., dict[str, Any]]) -> Callable[..., RawPayload]:
    """Factory for :class:`RawPayload` envelopes ready for the transform stage."""

    def _make(
        *,
        city_slug: str = "london",
        record_type: str = "observation",
        endpoint: str = "https://api.open-meteo.com/v1/forecast",
        requested_at_utc: datetime | None = None,
        start: datetime = DEFAULT_START,
        hours: int = 48,
        payload: Mapping[str, Any] | None = None,
        **payload_kwargs: Any,
    ) -> RawPayload:
        return RawPayload(
            city=get_city(city_slug),
            record_type=record_type,
            endpoint=endpoint,
            params={},
            payload=payload if payload is not None else make_hourly_payload(start=start, hours=hours, **payload_kwargs),
            requested_at_utc=requested_at_utc or datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        )

    return _make


@pytest.fixture()
def canonical_frame(make_raw_payload: Callable[..., RawPayload]) -> pl.DataFrame:
    """A small canonical frame (2 cities, 2 record types, 24 hours)."""
    payloads = []
    for slug in ("london", "tokyo"):
        payloads.append(make_raw_payload(city_slug=slug, record_type="observation"))
        payloads.append(make_raw_payload(city_slug=slug, record_type="forecast"))
    return transform_many(payloads)


@pytest.fixture()
def analytics_lake(tmp_path: Path, make_raw_payload: Callable[..., RawPayload]) -> Path:
    """A real on-disk lake with a known, easily verified temperature series."""
    from src.load import load_hourly

    payloads = [
        make_raw_payload(
            city_slug=slug,
            record_type="observation",
            start=datetime(2026, 10, 1, 0),
            hours=48,
            series_overrides={"temperature_2m": lambda index: float(index)},
        )
        for slug in ("london", "tokyo")
    ]
    frame = transform_many(payloads, ingested_at_utc=datetime(2026, 10, 3, tzinfo=timezone.utc))
    load_hourly(frame, tmp_path)
    return tmp_path


# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------


def query_params(request: requests.PreparedRequest) -> dict[str, str]:
    """Return the query string of a prepared request as a flat dict."""
    parsed = parse_qs(urlparse(str(request.url)).query)
    return {key: values[0] for key, values in parsed.items() if values}


def json_response(payload: Mapping[str, Any], status: int = 200) -> tuple[int, dict[str, str], str]:
    """Build a (status, headers, body) tuple for responses callbacks."""
    body = json.dumps(payload)
    return status, {"Content-Type": "application/json"}, body
