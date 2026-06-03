"""Shared pytest fixtures for the data-layer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    """An isolated cache root so tests never touch the repo's data/cache."""
    return tmp_path / "cache"
