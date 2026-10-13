"""Transform layer: normalise Open-Meteo JSON into a typed hourly frame.

Responsibilities
----------------
* Flatten the nested `hourly` block (a dict of parallel arrays) into tabular rows.
* Convert local wall-clock timestamps to genuine UTC instants.
* Coerce nulls/unparseable values, enforce the canonical dtype schema and apply the
  documented fill defaults.
* Return an empty, correctly typed frame instead of raising when a payload simply
  contains no usable hours.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import polars as pl

from src.config import (
    COLUMN_ORDER,
    DEDUP_KEYS,
    FILL_DEFAULTS,
    HOURLY_SCHEMA,
    INTEGER_MEASUREMENT_COLUMNS,
    MEASUREMENT_COLUMNS,
    RECORD_TYPE_OBSERVATION,
)
from src.extract import RawPayload

LOGGER = logging.getLogger(__name__)

#: Schema used while building frames from Python objects. The two timestamp columns
#: start out timezone-naive (the values are already UTC wall clock) and are labelled
#: UTC immediately afterwards, which is the shape polars handles most predictably.
BUILD_SCHEMA: dict[str, pl.DataType] = {
    name: (pl.Datetime("us") if dtype == pl.Datetime("us", "UTC") else dtype)
    for name, dtype in HOURLY_SCHEMA.items()
}


class TransformError(ValueError):
    """Raised when a payload is structurally unusable."""


# --------------------------------------------------------------------------------------
# Primitive coercion helpers
# --------------------------------------------------------------------------------------


def parse_utc_timestamp(value: Any, utc_offset_seconds: int = 0) -> datetime:
    """Parse an Open-Meteo timestamp into a timezone-aware UTC datetime.

    Parameters
    ----------
    value:
        ISO-8601 string (with or without offset) or an existing datetime.
    utc_offset_seconds:
        Offset of the zone the naive timestamps are expressed in. Always 0 for this
        pipeline because the API is queried with timezone=UTC, but handled
        explicitly so the function is correct for any requested zone.

    Raises
    ------
    TransformError
        If the value is empty or cannot be parsed.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise TransformError("empty timestamp value")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TransformError(f"unparseable timestamp {value!r}") from exc
    else:
        raise TransformError(f"unsupported timestamp type {type(value).__name__}")

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone(timedelta(seconds=int(utc_offset_seconds))))
    return moment.astimezone(timezone.utc)


def to_naive_utc(moment: datetime) -> datetime:
    """Return the UTC wall-clock rendering of `moment` (tzinfo stripped)."""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def coerce_float(value: Any) -> float | None:
    """Best-effort float conversion; anything unusable becomes None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def coerce_int(value: Any) -> int | None:
    """Best-effort integer conversion; anything unusable becomes None."""
    number = coerce_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return None


def _series(hourly: Mapping[str, Any], name: str) -> Sequence[Any]:
    values = hourly.get(name)
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TransformError(f"hourly.{name} is not a list of values")
    return values


def _value_at(values: Sequence[Any], index: int) -> Any:
    """Index into a parallel array, tolerating short arrays."""
    try:
        return values[index]
    except IndexError:
        return None


# --------------------------------------------------------------------------------------
# Frame construction
# --------------------------------------------------------------------------------------


def empty_hourly_frame() -> pl.DataFrame:
    """Return a zero-row frame carrying the canonical schema."""
    return pl.DataFrame(schema=HOURLY_SCHEMA)


def _build_columns(raw: RawPayload, ingested_at_utc: datetime) -> tuple[dict[str, list[Any]], int]:
    payload = raw.payload
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise TransformError(f"{raw.label}: payload has no usable 'hourly' block")

    times = _series(hourly, "time")
    if len(times) == 0:
        raise TransformError(f"{raw.label}: 'hourly.time' is empty")

    offset = coerce_int(payload.get("utc_offset_seconds")) or 0
    elevation = coerce_float(payload.get("elevation"))
    city = raw.city
    ingested_naive = to_naive_utc(ingested_at_utc)

    measurement_series = {name: _series(hourly, name) for name in MEASUREMENT_COLUMNS}

    columns: dict[str, list[Any]] = {name: [] for name in COLUMN_ORDER}
    skipped = 0

    for index, stamp in enumerate(times):
        try:
            moment = to_naive_utc(parse_utc_timestamp(stamp, offset))
        except TransformError as exc:
            skipped += 1
            LOGGER.debug("%s: skipping unparseable timestamp at index %d (%s)", raw.label, index, exc)
            continue

        columns["city"].append(city.name)
        columns["city_slug"].append(city.slug)
        columns["latitude"].append(float(city.latitude))
        columns["longitude"].append(float(city.longitude))
        columns["record_type"].append(raw.record_type)
        columns["source_endpoint"].append(raw.endpoint)
        columns["timestamp_utc"].append(moment)
        columns["date_utc"].append(moment.date())
        columns["hour_utc"].append(moment.hour)
        for name, values in measurement_series.items():
            value = _value_at(values, index)
            if name in INTEGER_MEASUREMENT_COLUMNS:
                columns[name].append(coerce_int(value))
            else:
                columns[name].append(coerce_float(value))
        columns["api_utc_offset_seconds"].append(offset)
        columns["api_elevation_m"].append(elevation)
        columns["ingested_at_utc"].append(ingested_naive)

    return columns, skipped


def enforce_schema(frame: pl.DataFrame) -> pl.DataFrame:
    """Add missing columns, cast every column to its canonical dtype and reorder.

    Casts are non-strict on purpose: a value that cannot be represented becomes null
    rather than aborting an entire run.
    """
    present = set(frame.columns)
    missing = [name for name in COLUMN_ORDER if name not in present]
    if missing:
        LOGGER.debug("Adding %d missing canonical column(s): %s", len(missing), ", ".join(missing))
        frame = frame.with_columns(
            [pl.lit(None, dtype=BUILD_SCHEMA[name]).alias(name) for name in missing]
        )
    frame = frame.select(list(COLUMN_ORDER))
    frame = frame.with_columns(
        [pl.col(name).cast(HOURLY_SCHEMA[name], strict=False) for name in COLUMN_ORDER]
    )
    return frame


def apply_fill_defaults(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the documented fill defaults for accumulator-style columns."""
    if frame.height == 0:
        return frame
    return frame.with_columns(
        [pl.col(name).fill_null(value) for name, value in FILL_DEFAULTS.items()]
    )


def drop_future_observations(frame: pl.DataFrame, raw: RawPayload) -> pl.DataFrame:
    """Remove hours that lie after the request instant from an observation payload.

    The forecast endpoint is queried with a `past_days` window to obtain measured
    values for recent hours, but the same response also contains the modelled hours
    that follow. Labelling those as observations would silently turn forecasts into
    "measurements", so anything after the moment of the request is dropped. Archive
    responses are entirely historical and pass through untouched.
    """
    if raw.record_type != RECORD_TYPE_OBSERVATION or frame.height == 0:
        return frame
    cutoff = raw.requested_at_utc.astimezone(timezone.utc)
    kept = frame.filter(pl.col("timestamp_utc") <= cutoff)
    dropped = frame.height - kept.height
    if dropped:
        LOGGER.info(
            "%s: dropped %d modelled hour(s) that follow the request instant",
            raw.label,
            dropped,
        )
    return kept


def clean_and_cast(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop unusable rows, enforce the schema, drop duplicates and sort."""
    if frame.height == 0:
        return empty_hourly_frame()
    frame = frame.filter(
        pl.col("timestamp_utc").is_not_null() & pl.col("city_slug").is_not_null()
    )
    frame = apply_fill_defaults(enforce_schema(frame))
    frame = frame.unique(subset=list(DEDUP_KEYS), keep="last", maintain_order=True)
    return frame.sort(["city_slug", "record_type", "timestamp_utc"])


def transform_payload(raw: RawPayload, ingested_at_utc: datetime | None = None) -> pl.DataFrame:
    """Normalise one API payload into the canonical hourly schema.

    Returns an empty frame (never raises) when the payload contains no usable hours,
    so that a single bad city cannot abort a run.
    """
    ingested = ingested_at_utc or datetime.now(tz=timezone.utc)
    try:
        columns, skipped = _build_columns(raw, ingested)
    except TransformError as exc:
        LOGGER.warning("Transforming %s produced no rows: %s", raw.label, exc)
        return empty_hourly_frame()

    if not columns["timestamp_utc"]:
        LOGGER.warning("%s: every timestamp was unparseable; nothing to load", raw.label)
        return empty_hourly_frame()

    frame = pl.DataFrame(columns, schema=BUILD_SCHEMA, strict=True)
    frame = frame.with_columns(
        [
            pl.col("timestamp_utc").dt.replace_time_zone("UTC"),
            pl.col("ingested_at_utc").dt.replace_time_zone("UTC"),
            pl.col("date_utc").cast(pl.Date),
        ]
    )
    frame = drop_future_observations(frame, raw)
    frame = clean_and_cast(frame)
    LOGGER.info(
        "Transformed %s: %d row(s), %d unparseable timestamp(s) skipped",
        raw.label,
        frame.height,
        skipped,
    )
    return frame


def transform_many(
    payloads: Sequence[RawPayload], ingested_at_utc: datetime | None = None
) -> pl.DataFrame:
    """Transform several payloads into a single canonical frame."""
    ingested = ingested_at_utc or datetime.now(tz=timezone.utc)
    frames = [transform_payload(raw, ingested) for raw in payloads]
    usable = [frame for frame in frames if frame.height > 0]
    if not usable:
        LOGGER.warning("No payload produced any rows; returning an empty frame")
        return empty_hourly_frame()
    combined = pl.concat(usable, how="diagonal_relaxed")
    combined = clean_and_cast(combined)
    LOGGER.info(
        "Transformed %d payload(s) into %d row(s) across %d city/city-type group(s)",
        len(payloads),
        combined.height,
        combined.select(["city_slug", "record_type"]).unique().height,
    )
    return combined


def describe_frame(frame: pl.DataFrame) -> dict[str, Any]:
    """Return small, log-friendly summary statistics for a frame."""
    if frame.height == 0:
        return {"rows": 0}
    return {
        "rows": frame.height,
        "cities": int(frame["city_slug"].n_unique()),
        "record_types": sorted(frame["record_type"].unique().to_list()),
        "first_timestamp_utc": frame["timestamp_utc"].min().isoformat(),
        "last_timestamp_utc": frame["timestamp_utc"].max().isoformat(),
        "temperature_nulls": int(frame["temperature_2m"].null_count()),
    }
