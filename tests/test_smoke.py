"""Smoke test — proves the package imports and CI runs green."""

def test_package_imports():
    import seizure_onset_detector  # noqa: F401

def test_python_version():
    import sys
    assert sys.version_info >= (3, 11)