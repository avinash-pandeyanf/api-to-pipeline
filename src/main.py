from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from src import __version__
from src.analytics import AnalyticsError, AnalyticsReport, run_analytics
from src.config import (
    DEFAULT_DATA_DIR,
    ConfigurationError,
    ensure_timezone_database,
    get_city,
    resolve_cities,
    utc_today,
)
from src.extract import ExtractionReport, HttpClient, HttpConfig, extract_many
from src.load import LoadReport, load_hourly
from src.transform import describe_frame, transform_many

LOGGER = logging.getLogger("src.main")

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

MAX_BACKFILL_DAYS = 366

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def configure_logging(level: str) -> None:
    """Configure root logging once, in a GitHub-Actions-friendly format."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT, datefmt=DATE_FORMAT)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="weather-etl",
        description="Extract Open-Meteo hourly weather data and land it as partitioned Parquet.",
    )
    parser.add_argument(
        "--backfill",
        type=int,
        metavar="DAYS",
        help=f"process the trailing DAYS days (1-{MAX_BACKFILL_DAYS}), oldest first",
    )
    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        metavar="YYYY-MM-DD",
        help="process one specific UTC date; repeat the flag for several dates",
    )
    parser.add_argument(
        "--cities",
        metavar="SLUG,SLUG",
        help="comma-separated city slugs (default: every configured city)",
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        default=str(DEFAULT_DATA_DIR),
        help="data lake root; hourly data lands in PATH/hourly (default: %(default)s)",
    )
    parser.add_argument("--skip-forecast", action="store_true", help="do not fetch forecasts")
    parser.add_argument(
        "--skip-observations", action="store_true", help="do not fetch measured observations"
    )
    parser.add_argument("--skip-analytics", action="store_true", help="skip the DuckDB stage")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any single request failed",
    )
    parser.add_argument(
        "--report-json", metavar="PATH", help="write a machine-readable run report to PATH"
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="HTTP attempts per request")
    parser.add_argument(
        "--request-timeout", type=float, default=30.0, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="logging verbosity (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_date(value: str) -> date:
    """Parse an ISO-8601 date, raising a user-facing error when malformed."""
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"invalid --date value {value!r}; expected YYYY-MM-DD") from exc


def resolve_target_dates(args: argparse.Namespace, today: date | None = None) -> list[date]:
    """Work out which UTC dates this run should cover."""
    if args.dates and args.backfill is not None:
        raise ConfigurationError("--date and --backfill are mutually exclusive")
    reference = today or utc_today()
    if args.dates:
        dates = sorted({parse_date(value) for value in args.dates})
        future = [d for d in dates if d > reference]
        if future:
            raise ConfigurationError(
                f"cannot process future date(s): {', '.join(d.isoformat() for d in future)}"
            )
        return dates
    if args.backfill is not None:
        if not 1 <= args.backfill <= MAX_BACKFILL_DAYS:
            raise ConfigurationError(f"--backfill must be between 1 and {MAX_BACKFILL_DAYS}")
        return [reference - timedelta(days=offset) for offset in range(args.backfill - 1, -1, -1)]
    return [reference]


def resolve_city_slugs(raw: str | None) -> tuple[str, ...] | None:
    """Split the --cities flag into validated, canonical slugs.

    Validation happens here (rather than only in the extractor) so a typo fails fast,
    before any HTTP request is issued.
    """
    if not raw:
        return None
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        raise ConfigurationError("--cities was provided but contained no usable slug")
    return tuple(get_city(part).slug for part in parts)


@dataclass(slots=True)
class RunReport:
    """Structured record of one pipeline execution (also written as JSON in CI)."""

    started_at_utc: str
    data_dir: str
    cities: list[str] = field(default_factory=list)
    target_dates: list[str] = field(default_factory=list)
    version: str = __version__
    finished_at_utc: str | None = None
    extracted_payloads: int = 0
    extraction_errors: list[dict[str, str]] = field(default_factory=list)
    transformed_rows: int = 0
    loaded_rows: int = 0
    partitions_touched: int = 0
    analytics: dict[str, object] | None = None
    analytics_error: str | None = None
    transform_summary: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "data_dir": self.data_dir,
            "cities": self.cities,
            "target_dates": self.target_dates,
            "extracted_payloads": self.extracted_payloads,
            "extraction_errors": self.extraction_errors,
            "transformed_rows": self.transformed_rows,
            "loaded_rows": self.loaded_rows,
            "partitions_touched": self.partitions_touched,
            "transform_summary": self.transform_summary,
            "analytics": self.analytics,
            "analytics_error": self.analytics_error,
        }


def run(args: argparse.Namespace, today: date | None = None) -> tuple[int, RunReport]:
    """Execute the pipeline. Returns an exit code and the run report."""
    ensure_timezone_database()
    reference = today or utc_today()
    data_dir = Path(args.data_dir).expanduser().resolve()
    cities = resolve_cities(resolve_city_slugs(args.cities))
    target_dates = resolve_target_dates(args, today=reference)
    if args.skip_forecast and args.skip_observations:
        raise ConfigurationError("--skip-forecast and --skip-observations leave nothing to do")

    report = RunReport(
        started_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        data_dir=data_dir.as_posix(),
        cities=[city.slug for city in cities],
        target_dates=[d.isoformat() for d in target_dates],
    )

    LOGGER.info(
        "Run start | version=%s cities=%s dates=%s data_dir=%s",
        __version__,
        ",".join(report.cities),
        ",".join(report.target_dates),
        report.data_dir,
    )

    client = HttpClient(
        config=HttpConfig(max_attempts=args.max_attempts, timeout_seconds=args.request_timeout)
    )
    extraction: ExtractionReport = extract_many(
        client,
        cities,
        target_dates,
        today=reference,
        include_forecast=not args.skip_forecast,
        include_observation=not args.skip_observations,
    )
    report.extracted_payloads = extraction.succeeded
    report.extraction_errors = [{"target": target, "error": error} for target, error in extraction.errors]

    if not extraction.payloads:
        LOGGER.error("No payloads were retrieved; aborting before the load stage")
        report.finished_at_utc = datetime.now(tz=timezone.utc).isoformat()
        return EXIT_FAILURE, report

    frame = transform_many(extraction.payloads)
    report.transformed_rows = frame.height
    report.transform_summary = describe_frame(frame)

    load_report: LoadReport = load_hourly(frame, data_dir)
    report.loaded_rows = load_report.rows_written
    report.partitions_touched = load_report.files_touched

    if not args.skip_analytics:
        try:
            analytics_report: AnalyticsReport = run_analytics(data_dir)
            report.analytics = analytics_report.as_dict()
        except AnalyticsError as exc:
            LOGGER.warning("Analytics stage skipped: %s", exc)
            report.analytics_error = str(exc)

    report.finished_at_utc = datetime.now(tz=timezone.utc).isoformat()
    LOGGER.info(
        "Run finished | payloads=%d rows=%d files=%d errors=%d",
        report.extracted_payloads,
        report.loaded_rows,
        report.partitions_touched,
        len(report.extraction_errors),
    )

    if args.strict and report.extraction_errors:
        return EXIT_FAILURE, report
    if report.transformed_rows == 0:
        LOGGER.error("Nothing was transformed; treating the run as failed")
        return EXIT_FAILURE, report
    return EXIT_OK, report


def write_report(path: str | Path, report: RunReport) -> Path:
    """Persist the run report as indented JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote run report to %s", target.as_posix())
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        exit_code, report = run(args)
    except ConfigurationError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        LOGGER.error("Interrupted")
        return 130
    if args.report_json:
        write_report(args.report_json, report)
    return exit_code


if __name__ == "__main__":  # pragma: no cover - module execution guard
    sys.exit(main())
