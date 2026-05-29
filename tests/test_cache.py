import numpy as np
import pytest

from seizure_onset_detector import cache


@pytest.fixture
def synthetic_patient():
    return {
        "X": np.arange(12, dtype=np.float32).reshape(6, 2),
        "y": np.array([0, 0, 1, 1, 0, 1], dtype=np.int8),
        "start_times": np.arange(6, dtype=np.float64) * 0.5,
        "seizure_intervals": np.array([[100.0, 130.0], [400.0, 410.0]]),
    }


def test_save_load_roundtrip(tmp_path, synthetic_patient):
    cache.save_patient_cache("03", synthetic_patient, cache_dir=tmp_path)
    loaded = cache.load_patient_cache("03", cache_dir=tmp_path)

    assert np.array_equal(loaded["X"], synthetic_patient["X"])
    assert np.array_equal(loaded["y"], synthetic_patient["y"])
    assert np.array_equal(loaded["start_times"], synthetic_patient["start_times"])
    assert loaded["seizure_intervals"] == [(100.0, 130.0), (400.0, 410.0)]


def test_load_missing_returns_none(tmp_path):
    assert cache.load_patient_cache("99", cache_dir=tmp_path) is None


def test_stale_params_invalidates_cache(tmp_path, synthetic_patient, monkeypatch):
    cache.save_patient_cache("03", synthetic_patient, cache_dir=tmp_path)
    # Simulate the feature code changing after the cache was written.
    monkeypatch.setattr(
        cache, "_extraction_params", lambda: {"window_sec": 2.0, "feature_names": ["x"]}
    )
    assert cache.load_patient_cache("03", cache_dir=tmp_path) is None


def test_feature_code_change_invalidates_cache(tmp_path, synthetic_patient, monkeypatch):
    cache.save_patient_cache("03", synthetic_patient, cache_dir=tmp_path)
    # Same params, but the features module source hash differs (edited feature code).
    monkeypatch.setattr(cache, "_feature_code_hash", lambda: "different-hash")
    assert cache.load_patient_cache("03", cache_dir=tmp_path) is None


def test_build_or_load_uses_cache_without_extracting(tmp_path, synthetic_patient, monkeypatch):
    cache.save_patient_cache("03", synthetic_patient, cache_dir=tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("build_patient_features should not run on a cache hit")

    monkeypatch.setattr(cache, "build_patient_features", _boom)
    out = cache.build_or_load_patient("unused_dir", "03", cache_dir=tmp_path)
    assert np.array_equal(out["X"], synthetic_patient["X"])


def test_load_cohort_concatenates_patients(tmp_path, synthetic_patient, monkeypatch):
    cache.save_patient_cache("03", synthetic_patient, cache_dir=tmp_path)
    cache.save_patient_cache("04", synthetic_patient, cache_dir=tmp_path)
    monkeypatch.setattr(
        cache, "build_patient_features",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not extract")),
    )

    X, y, pids, times, intervals = cache.load_cohort_features(
        "unused_dir", ["03", "04"], cache_dir=tmp_path
    )
    n = len(synthetic_patient["y"])
    assert X.shape == (2 * n, 2)
    assert list(np.unique(pids)) == ["03", "04"]
    assert (pids == "03").sum() == n
    assert set(intervals) == {"03", "04"}
    assert intervals["03"] == [(100.0, 130.0), (400.0, 410.0)]
