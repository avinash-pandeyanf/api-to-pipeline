# Open-Meteo -> Parquet ETL Pipeline

A lightweight, fully automated **API-to-Parquet** pipeline: it pulls hourly weather
forecasts and measured observations for three major cities from the
[Open-Meteo](https://open-meteo.com/) public REST API (no API key required), cleans and
types the payloads, writes snappy-compressed Hive-partitioned Parquet, and uses **DuckDB**
to publish an analysis-ready summary. GitHub Actions refreshes everything every 6 hours.

Python 3.11-3.14 | requests + polars + pyarrow | DuckDB | pytest | 161 tests

---

## 1. What it does

| Stage | Module | Responsibility |
| --- | --- | --- |
| **Extract** | `src/extract.py` | Calls Open-Meteo's forecast and archive endpoints with exponential backoff, jitter, `Retry-After` support and per-city failure isolation. |
| **Transform** | `src/transform.py` | Flattens the nested `hourly` block, normalises nulls, converts every timestamp to a true UTC instant, and enforces the canonical dtype schema. |
| **Load** | `src/load.py` | Writes daily partitioned Parquet with **atomic** `os.replace` swaps and **idempotent** de-duplicating upserts. |
| **Analytics** | `src/analytics.py` | Runs DuckDB SQL straight over the lake to produce 24-hour rolling averages, daily min/max temperatures and `summary.parquet`. |
| **CLI** | `src/main.py` | Wires the stages together behind `argparse` (`--backfill`, `--date`, `--cities`, ...) and emits a JSON run report. |

## 2. Architecture

```
                        Open-Meteo public REST API (no API key)
             +--------------------------+---------------------------+
             |                                                      |
   api.open-meteo.com/v1/forecast                    archive-api.open-meteo.com/v1/archive
   (forecast + measured past_days window)            (ERA5 reanalysis, lags about 5 days)
             |                                                      |
             +--------------------------+---------------------------+
                                        |
                                 src/extract.py
                     retry/backoff . error isolation . envelope metadata
                                        |
                                 src/transform.py
              flatten JSON -> null handling -> dtype schema -> UTC instants
                                        |
                                  src/load.py
       atomic write (tmp + fsync + os.replace) . upsert . snappy . partitions
                                        v
   data/hourly/year=YYYY/month=MM/day=DD/<city>__<record_type>.parquet
                                        |
                                src/analytics.py  (DuckDB)
             rolling_24h.parquet . daily_metrics.parquet . summary.parquet
                                        v
                     data/analytics/*.parquet  -> committed by GitHub Actions
```

## 3. Repository layout

```
openmeteo-parquet-etl/
|-- .github/workflows/scheduled_etl.yml   # cron 0 */6 * * * + workflow_dispatch
|-- src/
|   |-- __init__.py
|   |-- config.py        # endpoints, city catalogue, canonical schema, paths
|   |-- extract.py       # HTTP client with retry/backoff + request builders
|   |-- transform.py     # JSON flattening, coercion, schema enforcement, UTC
|   |-- load.py          # atomic, partitioned, idempotent Parquet writes
|   |-- analytics.py     # DuckDB views + exports
|   |-- main.py          # argparse CLI entry point
|-- tests/
|   |-- conftest.py      # payload factories, lake fixtures, no-network guard
|   |-- test_extract.py  # retry policy, error handling, endpoint routing
|   |-- test_transform.py# edge cases: missing attrs, empty records, bad timestamps
|   |-- test_load.py     # partitioning, atomicity, idempotency, snappy
|   |-- test_analytics.py# DuckDB correctness on a fixture lake
|   |-- test_main.py     # CLI parsing + end-to-end runs with mocked HTTP
|-- data/                # generated (committed by CI)
|-- pyproject.toml       # packaging, pinned deps, pytest/coverage config
|-- requirements.txt     # flat pinned install list
|-- README.md
```

## 4. Data model

### 4.1 Canonical hourly schema

Every Parquet file has exactly these columns, in this order
(`src/config.py` -> `HOURLY_SCHEMA` is the single source of truth):

| Column | Type | Notes |
| --- | --- | --- |
| `city` / `city_slug` | String | Display name and stable key (`london`, `tokyo`, `new_york`). |
| `latitude` / `longitude` | Float64 | Coordinates used for the request. |
| `record_type` | String | `forecast` or `observation`. |
| `source_endpoint` | String | Which Open-Meteo endpoint produced the row. |
| `timestamp_utc` | Datetime(us, UTC) | **Timezone-aware** UTC instant - the partition key. |
| `date_utc` / `hour_utc` | Date / Int8 | Derived UTC date and hour. |
| `temperature_2m` | Float64 | Air temperature 2 m above ground, degrees C. |
| `relative_humidity_2m` | Float64 | Percent. |
| `apparent_temperature` | Float64 | Feels-like, degrees C. |
| `precipitation` / `rain` / `snowfall` | Float64 | Millimetres; filled with 0.0 when absent (see 4.3). |
| `weather_code` | Int16 | WMO code; -1 marks "unreported". |
| `cloud_cover` | Float64 | Percent. |
| `pressure_msl` / `surface_pressure` | Float64 | hPa. |
| `wind_speed_10m` / `wind_direction_10m` / `wind_gusts_10m` | Float64 | km/h and degrees. |
| `api_utc_offset_seconds` / `api_elevation_m` | Int32 / Float64 | Provenance from the API response. |
| `ingested_at_utc` | Datetime(us, UTC) | When this pipeline run processed the row. |

### 4.2 Partition layout

```
data/
|-- hourly/
|   |-- year=2026/month=10/day=12/
|   |   |-- london__forecast.parquet
|   |   |-- london__observation.parquet
|   |   |-- tokyo__forecast.parquet
|   |   |-- ...
|-- analytics/
|   |-- rolling_24h.parquet
|   |-- daily_metrics.parquet
|   |-- summary.parquet
```

Partition columns live **only in the directory names**, never duplicated inside the file,
so DuckDB reconstructs them for free:

```sql
SELECT city_slug, min(temperature_2m), max(temperature_2m)
FROM read_parquet('data/hourly/**/*.parquet', hive_partitioning = true)
GROUP BY ALL;
```

### 4.3 Null and missing-value policy

* **Measurements are never fabricated.** A missing temperature stays null.
* **Accumulators default to zero.** `precipitation`, `rain` and `snowfall`
  become 0.0 when the API omits them, because "no value" means "nothing fell".
* **`weather_code` defaults to -1**, an explicit "unreported" sentinel (WMO code 0
  means clear sky, so it cannot double as a null marker).
* Unparseable timestamps and structurally unusable payloads are dropped with a warning -
  one bad city never aborts a run.

### 4.4 Analytics outputs

| File | Grain | Highlights |
| --- | --- | --- |
| `rolling_24h.parquet` | city x record type x hour | Trailing 24-hour window (current hour + preceding 23) computed with a time-based `RANGE` window, so gaps never inflate the average: `temperature_rolling_24h_avg_c`, `_min_c`, `_max_c`, `precipitation_rolling_24h_mm`, `rolling_window_samples`. |
| `daily_metrics.parquet` | city x record type x UTC day | `temperature_min_c`, `temperature_max_c`, `temperature_avg_c`, `temperature_range_c`, `precipitation_total_mm`, `wind_gust_max_kmh`, `humidity_avg_pct`, and sample counts per record type. |
| `summary.parquet` | city x record type x UTC day | The daily metrics **joined** with the rolling statistics, warmest/coldest hour and a `generated_at_utc` audit stamp. This is the table to point BI tools at. |

Example queries:

```sql
-- Warmest and coldest hour per city and day
SELECT city_slug, observation_date, warmest_hour_utc, coldest_hour_utc, temperature_range_c
FROM read_parquet('data/analytics/summary.parquet')
ORDER BY observation_date DESC, temperature_range_c DESC;

-- Where did the rolling 24-hour average peak?
SELECT city_slug, record_type, max(temperature_rolling_24h_avg_c) AS peak_rolling_avg_c
FROM read_parquet('data/analytics/rolling_24h.parquet')
GROUP BY ALL;
```

## 5. Quickstart

### 5.1 Install

```bash
git clone https://github.com/example/openmeteo-parquet-etl.git
cd openmeteo-parquet-etl

python -m venv .venv
source .venv/bin/activate        # Windows: .venv/Scripts/activate

pip install -r requirements.txt   # runtime + test dependencies
# or, for an installable package with a console script:
pip install -e ".[dev]"
```

No API key, token or secret of any kind is required.

**Two platform notes.** The suite and the pipeline are verified on Python 3.11-3.14.
On Windows (and in slim container images) Python ships **no IANA timezone database**, so
two non-obvious dependencies are pinned for you: `tzdata` lets `zoneinfo` resolve
`UTC` for timezone-aware timestamps, and `pytz` lets DuckDB convert `TIMESTAMPTZ`
values. Missing either one produces a confusing low-level error, so the pipeline checks
for the timezone database up front and tells you what to install.

### 5.2 Run

```bash
# Today, all three cities: forecast + observations + analytics
python -m src.main

# Rebuild the last 7 days of observations
python -m src.main --backfill 7

# One specific historical day, one city
python -m src.main --date 2026-10-01 --cities london
```

### 5.3 Test

```bash
python -m pytest -q                       # 161 tests, no network access
python -m pytest -q -m "not integration"  # skip filesystem/DuckDB integration tests
python -m pytest --cov=src --cov-report=term-missing
```

The suite installs an autouse guard that makes any real socket connection fail, so a test
can never silently depend on the public internet.

## 6. CLI reference

```text
python -m src.main [options]

  --backfill DAYS         Process the trailing DAYS UTC days (1-366), oldest first.
  --date YYYY-MM-DD       Process one specific UTC date; repeat for several dates.
                          Mutually exclusive with --backfill.
  --cities SLUG,SLUG      City slugs (default: london,tokyo,new_york).
  --data-dir PATH         Data lake root (default: ./data, or $WEATHER_ETL_DATA_DIR).
  --skip-forecast         Do not fetch forecasts.
  --skip-observations     Do not fetch measured observations.
  --skip-analytics        Skip the DuckDB stage.
  --strict                Exit non-zero if any single request failed.
  --report-json PATH      Write a machine-readable run report.
  --max-attempts N        HTTP attempts per request (default 5).
  --request-timeout SEC   Per-request timeout (default 30).
  --log-level LEVEL       DEBUG | INFO | WARNING | ERROR (default INFO).
  --version               Print the version and exit.
```

**Exit codes:** 0 success, 1 runtime failure (no payloads, nothing transformed, or any
error under `--strict`), 2 configuration error (bad flag combination, unknown city,
malformed date, nothing left to fetch).

## 7. Orchestration (GitHub Actions)

`.github/workflows/scheduled_etl.yml` runs:

* **`schedule`:** `0 */6 * * *` - every 6 hours.
* **`workflow_dispatch`:** manual runs with `backfill_days`, `cities` and a
  `commit_data` toggle.

Two jobs keep responsibilities clean:

1. **`test`** - matrix over Python 3.11 and 3.12: install, `compileall` syntax gate,
   full `pytest` suite.
2. **`etl`** (`needs: test`) - install, run `python -m src.main`, append a file listing
   to the job summary, upload the lake as a 14-day artefact, then commit the Parquet with
   `stefanzweifel/git-auto-commit-action@v5` and push it back to the branch.

Publishing strategy - **both**, on purpose:

* `data/hourly/**/*.parquet` and `data/analytics/*.parquet` are committed back to the
  branch, giving the repository a queryable, versioned lake with no external storage.
* The same files are uploaded as a workflow artefact, so data survives even when
  `commit_data=false` or the push is rejected.

Operational details:

* `permissions: contents: write` is required by the auto-commit action.
* `concurrency` on `github.ref` with `cancel-in-progress: false` prevents two runs
  from racing on the same partition files.
* Commits use `[skip ci]` so the data refresh does not re-trigger workflows.
* Generated data is intentionally **not** git-ignored; only scratch output
  (`data/_reports/`, `*.tmp`) is excluded.

## 8. Design decisions

**Atomic writes.** Each partition is written to a uniquely named sibling temp file,
`fsync`-ed, then moved into place with `os.replace`. Readers see either the previous
complete file or the new complete one - never a truncated Parquet. A failed write removes
the temp file and leaves the old artefact intact.

**Idempotent upserts.** Re-running the same hour merges with the existing file and
de-duplicates on `(city_slug, record_type, timestamp_utc)`, so retries and overlapping
backfills are safe. Duplicate rows are counted in the run report.

**Genuine UTC instants.** The API is queried with `timezone=UTC` and payloads are parsed
against their declared `utc_offset_seconds`, so timestamps are true UTC instants, not
local wall-clock strings. DuckDB sessions are pinned with `SET TimeZone='UTC'` because
`CAST(timestamp AS DATE)` is evaluated in the session zone - a local-time connection
would silently shift every day boundary.

**Observations are never forecasts in disguise.** The forecast endpoint also returns
*modelled* hours after the request instant. Anything later than the moment of the request
is dropped from an observation payload instead of being mislabelled as a measurement.

**Time-based rolling windows.** The 24-hour window uses
`RANGE BETWEEN 86399 PRECEDING AND CURRENT ROW` over `epoch(timestamp_utc)`: exactly
the current hour plus the preceding 23, robust to gaps and DST.
`rolling_window_samples` exposes partial windows so consumers can filter them.

**Flat `src/` package.** The layout mirrors the requested spec
(`src/extract.py`, ...), runs as `python -m src.main`, and is also installable as the
`weather-etl` console script.

## 9. Extending

**Add a city** - append one line to `CITIES` in `src/config.py`:

```python
City("Berlin", "berlin", 52.5200, 13.4050, "Europe/Berlin"),
```

then pass `--cities berlin` (or just let the default run pick it up). No other change
is needed.

**Add a variable** - add it to `HOURLY_VARIABLES` and to `HOURLY_SCHEMA` with its
dtype (add it to `MEASUREMENT_COLUMNS`, and to `INTEGER_MEASUREMENT_COLUMNS` if it
is integral). Transform, load and DuckDB all pick it up automatically.

**Change the schedule** - edit the `cron` expression in the workflow.

## 10. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Analytics stage skipped: No Parquet files` | The lake is empty because every request failed; fix extraction first. |
| `exhausted N attempts` in the log | Open-Meteo was unreachable or rate-limiting; the run continues and reports the error. Use `--max-attempts` to tune. |
| `Required module 'pytz' failed to import` | DuckDB needs `pytz` to convert `TIMESTAMPTZ` into Python datetimes; it is pinned in `requirements.txt`. |
| `No time zone found with key UTC` or a polars `PanicException` | Python has no IANA timezone database (typical on Windows and slim images). Install `tzdata`: `pip install -r requirements.txt`. The pipeline now fails with this advice instead of a Rust panic. |
| Auto-commit step fails | Ensure `permissions: contents: write` and that the branch is not protected. Data is still uploaded as an artefact. |
| Day boundaries look shifted | Never query the lake from a local-timezone DuckDB session; use `src.analytics.connect()`, which pins UTC. |

## License

MIT.
