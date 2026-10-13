"""Extraction layer: pull hourly weather data from the Open-Meteo REST API.

Design notes
------------
* Every HTTP call goes through :class:`HttpClient`, which implements bounded
  exponential backoff with jitter and honours the `Retry-After` response header.
* The client accepts an injected `requests.Session` and `sleep` callable, so the
  retry policy is fully testable without real sockets or real waiting.
* Partial failures never abort a whole run: :func:`extract_many` records failures
  and keeps going, leaving it to the caller to decide how strict to be.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import requests

from src.config import (
    ARCHIVE_ENDPOINT,
    FORECAST_ENDPOINT,
    HOURLY_VARIABLES,
    MAX_PAST_DAYS,
    RECORD_TYPE_FORECAST,
    RECORD_TYPE_OBSERVATION,
    City,
    archive_cutoff,
    utc_today,
)

LOGGER = logging.getLogger(__name__)

#: HTTP status codes worth retrying. Everything else in the 4xx range is a
#: client error and retrying it would only burn the rate limit.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

USER_AGENT = "openmeteo-parquet-etl/1.0 (+https://github.com/example/openmeteo-parquet-etl)"


class ExtractionError(RuntimeError):
    """Raised when a payload cannot be retrieved after exhausting all retries."""


@dataclass(slots=True)
class HttpConfig:
    """Tunables for the retry policy."""

    max_attempts: int = 5
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0
    jitter_ratio: float = 0.25
    timeout_seconds: float = 30.0
    user_agent: str = USER_AGENT

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds must be > 0")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError("backoff_max_seconds must be >= backoff_base_seconds")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be within [0, 1]")


def build_session(config: HttpConfig | None = None) -> requests.Session:
    """Create a session with sane defaults (connection reuse + JSON headers)."""
    cfg = config or HttpConfig()
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent, "Accept": "application/json"})
    return session


@dataclass(slots=True)
class HttpClient:
    """Thin JSON-over-HTTP client with bounded exponential backoff.

    Parameters
    ----------
    config:
        Retry/backoff tunables.
    session:
        Injectable transport. A new :class:`requests.Session` is created when omitted.
    sleep:
        Injectable sleeper, so tests can assert on the backoff schedule instantly.
    random_source:
        Injectable RNG used for jitter, making the schedule deterministic in tests.
    """

    config: HttpConfig = field(default_factory=HttpConfig)
    session: requests.Session | None = None
    sleep: Callable[[float], None] = time.sleep
    random_source: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = build_session(self.config)

    # -- retry helpers -----------------------------------------------------------------

    def _compute_backoff(self, attempt: int) -> float:
        """Return the sleep duration before `attempt` (1-based) is retried."""
        raw = self.config.backoff_base_seconds * (2 ** (attempt - 1))
        capped = min(raw, self.config.backoff_max_seconds)
        if self.config.jitter_ratio <= 0:
            return capped
        return capped + self.random_source.uniform(0.0, capped * self.config.jitter_ratio)

    def _retry_after_seconds(self, response: requests.Response) -> float | None:
        """Parse a `Retry-After` header expressed in delta-seconds."""
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            seconds = float(raw.strip())
        except (TypeError, ValueError):
            return None
        if seconds < 0:
            return None
        return min(seconds, self.config.backoff_max_seconds)

    @staticmethod
    def _excerpt(response: requests.Response, limit: int = 300) -> str:
        try:
            body = response.text
        except Exception:  # pragma: no cover - defensive
            return "<unreadable response body>"
        body = " ".join(body.split())
        return body[:limit]

    # -- public API --------------------------------------------------------------------

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        context: str = "",
    ) -> Any:
        """GET `url` and decode JSON, retrying transient failures.

        Raises
        ------
        ExtractionError
            When every attempt fails, or the server returns a non-retryable
            status code, or the body is not valid JSON.
        """
        assert self.session is not None  # guaranteed by __post_init__
        label = context or url
        last_error: str = "unknown error"

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.session.get(
                    url, params=dict(params or {}), timeout=self.config.timeout_seconds
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.config.max_attempts:
                    break
                delay = self._compute_backoff(attempt)
                LOGGER.warning(
                    "GET %s failed on attempt %d/%d (%s); retrying in %.2fs",
                    label,
                    attempt,
                    self.config.max_attempts,
                    last_error,
                    delay,
                )
                self.sleep(delay)
                continue

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise ExtractionError(
                        f"{label}: response was not valid JSON ({exc})"
                    ) from exc

            last_error = f"HTTP {status}: {self._excerpt(response)}"
            if status not in RETRYABLE_STATUS_CODES:
                raise ExtractionError(f"{label}: non-retryable {last_error}")

            if attempt == self.config.max_attempts:
                break

            delay = self._retry_after_seconds(response)
            if delay is None:
                delay = self._compute_backoff(attempt)
            LOGGER.warning(
                "GET %s returned HTTP %d on attempt %d/%d; retrying in %.2fs",
                label,
                status,
                attempt,
                self.config.max_attempts,
                delay,
            )
            self.sleep(delay)

        raise ExtractionError(
            f"{label}: exhausted {self.config.max_attempts} attempts; last error: {last_error}"
        )


# --------------------------------------------------------------------------------------
# Payload envelope
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawPayload:
    """A raw API response plus the metadata the transform stage needs."""

    city: City
    record_type: str
    endpoint: str
    params: Mapping[str, Any]
    payload: Mapping[str, Any]
    requested_at_utc: datetime

    @property
    def label(self) -> str:
        return f"{self.city.slug}/{self.record_type}"


@dataclass(slots=True)
class ExtractionReport:
    """Outcome of an :func:`extract_many` call."""

    payloads: list[RawPayload] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.payloads)

    @property
    def succeeded(self) -> int:
        return len(self.payloads)

    @property
    def failed(self) -> int:
        return len(self.errors)


# --------------------------------------------------------------------------------------
# Request builders
# --------------------------------------------------------------------------------------


def _common_params(city: City) -> dict[str, Any]:
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }


def build_forecast_params(city: City, forecast_days: int = 2, past_days: int = 1) -> dict[str, Any]:
    """Query parameters for the forward-looking forecast endpoint."""
    if forecast_days < 1:
        raise ValueError("forecast_days must be >= 1")
    if not 0 <= past_days <= MAX_PAST_DAYS:
        raise ValueError(f"past_days must be within [0, {MAX_PAST_DAYS}]")
    params = _common_params(city)
    params.update({"forecast_days": forecast_days, "past_days": past_days})
    return params


def build_archive_params(city: City, start_date: date, end_date: date) -> dict[str, Any]:
    """Query parameters for the historical reanalysis endpoint."""
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    params = _common_params(city)
    params.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})
    return params


def recent_past_days(target_date: date, today: date) -> int:
    """Number of `past_days` needed to cover `target_date` from the forecast endpoint."""
    delta = (today - target_date).days
    return max(0, min(delta, MAX_PAST_DAYS))


def select_observation_endpoint(target_date: date, today: date | None = None) -> str:
    """Choose the endpoint that can actually serve `target_date`.

    Dates older than the archive lag come from the ERA5 archive; fresher dates come
    from the forecast endpoint's measured `past_days` window.
    """
    reference = today or utc_today()
    return ARCHIVE_ENDPOINT if target_date <= archive_cutoff(reference) else FORECAST_ENDPOINT


# --------------------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------------------


def _validate_payload(label: str, data: Any) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ExtractionError(f"{label}: expected a JSON object, got {type(data).__name__}")
    if data.get("error"):
        reason = data.get("reason") or data.get("reason") or "unknown API error"
        raise ExtractionError(f"{label}: API reported an error: {reason}")
    if "hourly" not in data:
        raise ExtractionError(f"{label}: response is missing the 'hourly' block")
    return data


def fetch_forecast(client: HttpClient, city: City, forecast_days: int = 2, past_days: int = 1) -> RawPayload:
    """Fetch the hourly forecast for `city` (including recent measured hours)."""
    params = build_forecast_params(city, forecast_days=forecast_days, past_days=past_days)
    label = f"forecast/{city.slug}"
    LOGGER.info("Fetching %s forecast from %s", city.name, FORECAST_ENDPOINT)
    data = client.get_json(FORECAST_ENDPOINT, params, context=label)
    return RawPayload(
        city=city,
        record_type=RECORD_TYPE_FORECAST,
        endpoint=FORECAST_ENDPOINT,
        params=params,
        payload=_validate_payload(label, data),
        requested_at_utc=datetime.now(tz=timezone.utc),
    )


def fetch_observation(
    client: HttpClient,
    city: City,
    target_date: date,
    today: date | None = None,
) -> RawPayload:
    """Fetch measured hourly observations covering `target_date` for `city`."""
    reference = today or utc_today()
    endpoint = select_observation_endpoint(target_date, reference)
    if endpoint == ARCHIVE_ENDPOINT:
        params = build_archive_params(city, target_date, target_date)
        mode = "archive"
    else:
        params = build_forecast_params(
            city, forecast_days=1, past_days=recent_past_days(target_date, reference)
        )
        mode = "recent"
    label = f"observation/{city.slug}/{target_date.isoformat()}"
    LOGGER.info("Fetching %s observations (%s source) for %s", city.name, mode, target_date)
    data = client.get_json(endpoint, params, context=label)
    return RawPayload(
        city=city,
        record_type=RECORD_TYPE_OBSERVATION,
        endpoint=endpoint,
        params=params,
        payload=_validate_payload(label, data),
        requested_at_utc=datetime.now(tz=timezone.utc),
    )


def extract_city(
    client: HttpClient,
    city: City,
    target_date: date,
    *,
    today: date | None = None,
    include_forecast: bool = True,
    include_observation: bool = True,
    forecast_days: int = 2,
    past_days: int = 1,
) -> list[RawPayload]:
    """Fetch every requested record type for one city on one date."""
    payloads: list[RawPayload] = []
    if include_observation:
        payloads.append(fetch_observation(client, city, target_date, today=today))
    if include_forecast:
        payloads.append(fetch_forecast(client, city, forecast_days=forecast_days, past_days=past_days))
    return payloads


def extract_many(
    client: HttpClient,
    cities: Sequence[City],
    target_dates: Sequence[date],
    *,
    today: date | None = None,
    include_forecast: bool = True,
    include_observation: bool = True,
    forecast_days: int = 2,
    past_days: int = 1,
) -> ExtractionReport:
    """Fetch all city/date combinations, isolating failures per request.

    The forecast is date-independent, so it is fetched once per city (for the first
    requested date) instead of once per date.
    """
    report = ExtractionReport()
    reference = today or utc_today()
    dates = list(target_dates)

    for city in cities:
        if include_observation:
            for target_date in dates:
                try:
                    report.payloads.append(
                        fetch_observation(client, city, target_date, today=reference)
                    )
                except ExtractionError as exc:
                    LOGGER.error("Observation extraction failed for %s: %s", city.slug, exc)
                    report.errors.append((f"{city.slug}/{target_date.isoformat()}", str(exc)))

        if include_forecast:
            try:
                report.payloads.append(
                    fetch_forecast(client, city, forecast_days=forecast_days, past_days=past_days)
                )
            except ExtractionError as exc:
                LOGGER.error("Forecast extraction failed for %s: %s", city.slug, exc)
                report.errors.append((f"{city.slug}/forecast", str(exc)))

    LOGGER.info(
        "Extraction finished: %d payload(s) retrieved, %d failure(s)",
        report.succeeded,
        report.failed,
    )
    return report
