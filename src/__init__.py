"""Open-Meteo hourly weather -> partitioned Parquet ETL pipeline.

The package is deliberately flat (`src/extract.py`, `src/transform.py`, ...) so that it can be
executed both as a module (`python -m src.main`) and through the installed console script
(`weather-etl`).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
