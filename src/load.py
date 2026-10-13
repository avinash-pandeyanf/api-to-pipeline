"""Load layer: partitioned, atomic Parquet writes.

Guarantees
----------
* Every partition file is written to a sibling temporary file, flushed to disk and
  then swapped in with `os.replace` (atomic on POSIX and Windows). A crash can
  therefore never leave a half-written `.parquet` behind.
* Writes are idempotent: re-running the same hour merges with the existing file and
  de-duplicates on (city_slug, record_type, timestamp_utc) instead of appending
  duplicates.
* Partitioning follows the Hive convention (`dataset/year=YYYY/month=MM/day=DD/`)
  so DuckDB and Spark can prune and reconstruct the partition columns for free.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Sequence

import polars as pl

from src.config import (
    DATASET_HOURLY,
    DEDUP_KEYS,
    PARQUET_COMPRESSION,
)
from src.transform import clean_and_cast

LOGGER = logging.getLogger(__name__)

TEMP_SUFFIX = ".tmp"


class LoadError(RuntimeError):
    """Raised when a partition cannot be written or verified."""


@dataclass(slots=True)
class PartitionResult:
    """Bookkeeping for a single partition file."""

    path: Path
    rows_in_batch: int
    rows_existing: int
    rows_written: int
    rows_dropped: int
    created: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "rows_in_batch": self.rows_in_batch,
            "rows_existing": self.rows_existing,
            "rows_written": self.rows_written,
            "rows_dropped": self.rows_dropped,
            "created": self.created,
        }


@dataclass(slots=True)
class LoadReport:
    """Aggregate outcome of a load call."""

    partitions: list[PartitionResult] = field(default_factory=list)

    @property
    def rows_written(self) -> int:
        return sum(part.rows_written for part in self.partitions)

    @property
    def files(self) -> list[Path]:
        return [part.path for part in self.partitions]

    @property
    def files_touched(self) -> int:
        return len(self.partitions)

    def as_dict(self) -> dict[str, object]:
        return {
            "files_touched": self.files_touched,
            "rows_written": self.rows_written,
            "partitions": [part.as_dict() for part in self.partitions],
        }


def partition_dir(data_dir: Path | str, when: date, dataset: str = DATASET_HOURLY) -> Path:
    """Return the Hive-style directory holding the data for `when`."""
    return (
        Path(data_dir)
        / dataset
        / f"year={when.year:04d}"
        / f"month={when.month:02d}"
        / f"day={when.day:02d}"
    )


def partition_path(
    data_dir: Path | str,
    city_slug: str,
    record_type: str,
    when: date,
    dataset: str = DATASET_HOURLY,
) -> Path:
    """Return the deterministic file path for one city/record-type/day slice."""
    if not city_slug or not record_type:
        raise LoadError("city_slug and record_type must be non-empty")
    return partition_dir(data_dir, when, dataset) / f"{city_slug}__{record_type}.parquet"


def hourly_glob(data_dir: Path | str, dataset: str = DATASET_HOURLY) -> str:
    """Return the posix-style recursive glob DuckDB/polars should scan."""
    return (Path(data_dir) / dataset / "**" / "*.parquet").as_posix()


# --------------------------------------------------------------------------------------
# Atomic IO primitives
# --------------------------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    """Flush a file's contents to stable storage."""
    try:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:  # pragma: no cover - platform dependent
        LOGGER.debug("fsync on %s failed (ignored): %s", path, exc)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync; a no-op on platforms that refuse to open dirs."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - Windows does not support this
        pass
    finally:
        os.close(fd)


def atomic_write_parquet(
    frame: pl.DataFrame,
    path: Path | str,
    compression: str = PARQUET_COMPRESSION,
) -> Path:
    """Write `frame` to `path` atomically.

    The frame is first written to a uniquely named temporary file next to the target,
    flushed with fsync, and only then moved into place with `os.replace`. Readers
    therefore observe either the previous complete file or the new complete file,
    never a partial one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}{TEMP_SUFFIX}")
    try:
        frame.write_parquet(tmp, compression=compression, statistics=True)
        _fsync_file(tmp)
        os.replace(tmp, target)
        _fsync_directory(target.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def read_partition(path: Path | str) -> pl.DataFrame:
    """Read one partition file back into memory (empty frame when absent)."""
    target = Path(path)
    if not target.exists():
        return pl.DataFrame()
    return pl.read_parquet(target)


def verify_partition(path: Path | str, expected_rows: int | None = None) -> int:
    """Re-open a written partition and confirm it is readable and complete."""
    target = Path(path)
    if not target.exists():
        raise LoadError(f"partition {target} was not created")
    try:
        frame = pl.read_parquet(target)
    except Exception as exc:
        raise LoadError(f"partition {target} is not readable: {exc}") from exc
    if expected_rows is not None and frame.height != expected_rows:
        raise LoadError(
            f"partition {target} holds {frame.height} rows, expected {expected_rows}"
        )
    return frame.height


# --------------------------------------------------------------------------------------
# Upsert + orchestration
# --------------------------------------------------------------------------------------


def upsert_partition(
    frame: pl.DataFrame,
    path: Path | str,
    compression: str = PARQUET_COMPRESSION,
) -> PartitionResult:
    """Merge `frame` into the partition at `path` and write it atomically."""
    target = Path(path)
    batch_rows = frame.height
    existing = read_partition(target)
    existing_rows = existing.height

    if existing_rows:
        combined = pl.concat([existing, frame], how="diagonal_relaxed")
    else:
        combined = frame

    combined = clean_and_cast(combined)
    combined = combined.unique(subset=list(DEDUP_KEYS), keep="last", maintain_order=True)
    combined = combined.sort(list(DEDUP_KEYS))
    rows_written = combined.height

    atomic_write_parquet(combined, target, compression=compression)
    written = verify_partition(target, expected_rows=rows_written)
    if written != rows_written:  # pragma: no cover - verify_partition already raises
        raise LoadError(f"post-write verification failed for {target}")

    result = PartitionResult(
        path=target,
        rows_in_batch=batch_rows,
        rows_existing=existing_rows,
        rows_written=rows_written,
        rows_dropped=max(0, batch_rows + existing_rows - rows_written),
        created=existing_rows == 0,
    )
    LOGGER.info(
        "Wrote %s (%d row(s): %d new batch + %d existing, %d duplicate(s) dropped)",
        target.as_posix(),
        rows_written,
        batch_rows,
        existing_rows,
        result.rows_dropped,
    )
    return result


def _iter_partitions(frame: pl.DataFrame) -> Sequence[tuple[tuple[date, str, str], pl.DataFrame]]:
    groups = frame.partition_by(["date_utc", "city_slug", "record_type"], as_dict=True)
    partitions: list[tuple[tuple[date, str, str], pl.DataFrame]] = []
    for key, subset in groups.items():
        parts = key if isinstance(key, tuple) else (key,)
        when_raw, city_slug, record_type = parts
        when = when_raw if isinstance(when_raw, date) else date.fromisoformat(str(when_raw))
        partitions.append(((when, str(city_slug), str(record_type)), subset))
    return partitions


def load_hourly(
    frame: pl.DataFrame,
    data_dir: Path | str,
    dataset: str = DATASET_HOURLY,
    compression: str = PARQUET_COMPRESSION,
) -> LoadReport:
    """Write a canonical hourly frame into its daily partitions.

    Parameters
    ----------
    frame:
        Canonical frame produced by :func:`src.transform.transform_many`.
    data_dir:
        Root of the data lake (the directory that contains `hourly/`).
    """
    report = LoadReport()
    if frame.height == 0:
        LOGGER.warning("Nothing to load: the frame has no rows")
        return report

    frame = clean_and_cast(frame)
    for (when, city_slug, record_type), subset in _iter_partitions(frame):
        target = partition_path(data_dir, city_slug, record_type, when, dataset)
        report.partitions.append(
            upsert_partition(subset, target, compression=compression)
        )

    LOGGER.info(
        "Load finished: %d file(s) touched, %d row(s) persisted",
        report.files_touched,
        report.rows_written,
    )
    return report
