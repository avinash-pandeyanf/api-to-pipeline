"""Guard rails for packaging metadata, environment assumptions and CI wiring.

The bug these tests exist for: the pipeline needs the IANA timezone database (through
`zoneinfo`) to materialise UTC timestamps, but `tzdata` was never declared as a
dependency. It happened to be pre-installed on the development machine, so 151 tests
passed there while 11 failed for anyone on a clean Windows environment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from src.config import ConfigurationError, ensure_timezone_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "scheduled_etl.yml"

#: Dependencies that are implicit at runtime and therefore easy to forget.
REQUIRED_RUNTIME_DEPENDENCIES = ("duckdb", "polars", "pyarrow", "requests", "pytz", "tzdata")

#: Python versions the project claims to support (floor plus CI ceiling).
SUPPORTED_PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")


def read_pyproject() -> dict[str, Any]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def read_requirements() -> list[str]:
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def read_workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_requirements_pins_every_runtime_dependency():
    requirements = read_requirements()
    for name in REQUIRED_RUNTIME_DEPENDENCIES:
        pinned = [req for req in requirements if req.lower().startswith(name.lower() + "==")]
        assert pinned, f"{name} must be pinned in requirements.txt (found: {requirements})"


def test_pyproject_declares_the_same_runtime_dependencies():
    declared = {
        dependency.split("==")[0].strip().lower()
        for dependency in read_pyproject()["project"]["dependencies"]
    }
    missing = set(REQUIRED_RUNTIME_DEPENDENCIES) - declared
    assert not missing, f"pyproject.toml is missing runtime dependencies: {sorted(missing)}"


def test_every_runtime_dependency_is_exactly_pinned():
    for dependency in read_pyproject()["project"]["dependencies"]:
        assert "==" in dependency, f"{dependency} is not pinned to an exact version"


def test_requirements_and_pyproject_agree_on_versions():
    declared = {
        dependency.split("==")[0].strip().lower(): dependency.split("==")[1].strip()
        for dependency in read_pyproject()["project"]["dependencies"]
    }
    for requirement in read_requirements():
        name, _, version = requirement.partition("==")
        key = name.strip().lower()
        if key in declared:
            assert declared[key] == version, f"{key} version differs between the two files"


def test_python_floor_is_declared():
    assert read_pyproject()["project"]["requires-python"] == ">=3.11"


def test_timezone_database_is_available():
    """The exact failure mode that broke 11 tests on the user's clean environment."""
    ensure_timezone_database()


def test_missing_timezone_database_reports_an_actionable_error(monkeypatch):
    import src.config as config_module

    def explode(_key: str) -> None:
        raise config_module.ZoneInfoNotFoundError("no database")

    monkeypatch.setattr(config_module, "ZoneInfo", explode)
    with pytest.raises(ConfigurationError, match="tzdata"):
        config_module.ensure_timezone_database()


def test_workflow_declares_schedule_and_manual_triggers():
    workflow = read_workflow()
    assert 'cron: "0 */6 * * *"' in workflow
    assert "workflow_dispatch:" in workflow


def test_workflow_publishes_data_back_to_the_repository():
    workflow = read_workflow()
    assert "stefanzweifel/git-auto-commit-action@v5" in workflow
    assert "permissions:" in workflow and "contents: write" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_ci_matrix_covers_every_supported_python_version():
    workflow = read_workflow()
    for version in SUPPORTED_PYTHON_VERSIONS:
        assert f'"{version}"' in workflow, f"CI matrix does not cover Python {version}"
