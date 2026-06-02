"""Smoke test: package imports and version is set."""

import cascade


def test_package_imports() -> None:
    assert hasattr(cascade, "__version__")
    assert isinstance(cascade.__version__, str)
