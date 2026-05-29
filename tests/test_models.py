import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from seizure_onset_detector.evaluate import leave_one_patient_out_cv
from seizure_onset_detector.models import build_logistic, build_random_forest, build_xgboost

try:
    import xgboost  # noqa: F401
    _HAS_XGBOOST = True
except Exception:  # native lib can fail to load (e.g. missing OpenMP runtime), not just ImportError
    _HAS_XGBOOST = False

requires_xgboost = pytest.mark.skipif(
    not _HAS_XGBOOST, reason="xgboost runtime unavailable (e.g. missing libomp)"
)


def test_build_logistic_uses_robustscaler_and_is_unfitted():
    pipe = build_logistic()
    assert isinstance(pipe.named_steps["scaler"], RobustScaler)
    assert not hasattr(pipe.named_steps["lr"], "coef_")  # returned unfitted


def test_build_random_forest_is_parallel_and_unfitted():
    rf = build_random_forest()
    assert rf.n_jobs == -1
    assert not hasattr(rf, "estimators_")  # returned unfitted


@requires_xgboost
def test_build_xgboost_is_unfitted_and_supports_predict_proba():
    clf = build_xgboost()
    assert clf.n_estimators == 200
    assert not hasattr(clf, "_Booster")  # returned unfitted
    assert hasattr(clf, "predict_proba")  # usable by cross_val_predict


def _toy_cohort():
    rng = np.random.default_rng(0)
    X, y, pids, times, seiz = [], [], [], [], {}
    for p, onset in [("01", 100.0), ("02", 150.0), ("03", 80.0)]:
        t = np.arange(0.0, 300.0, 0.5)
        lab = ((t >= onset) & (t < onset + 20)).astype(int)
        feat = lab + rng.normal(0, 0.3, size=len(t))  # label-separable feature
        X += list(feat)
        y += list(lab)
        pids += [p] * len(t)
        times += list(t)
        seiz[p] = [(onset, onset + 20)]
    return (
        np.asarray(X).reshape(-1, 1), np.asarray(y),
        np.asarray(pids), np.asarray(times), seiz,
    )


def test_lopocv_returns_one_row_per_patient_with_event_auprc():
    X, y, pids, times, seiz = _toy_cohort()
    m, c, em, ec = leave_one_patient_out_cv(
        X, y, pids, build_logistic(),
        start_times=times, seizure_intervals_by_patient=seiz, hop_sec=0.5,
    )
    assert {r["patient"] for r in m} == {"01", "02", "03"}
    assert {r["patient"] for r in em} == {"01", "02", "03"}
    assert all("event_auprc" in r for r in em)


@requires_xgboost
def test_lopocv_runs_with_xgboost():
    # XGBoost is sklearn-compatible, so it flows through cross_val_predict like the rest.
    X, y, pids, times, seiz = _toy_cohort()
    est = build_xgboost()
    m, c, em, ec = leave_one_patient_out_cv(
        X, y, pids, est,
        start_times=times, seizure_intervals_by_patient=seiz, hop_sec=0.5,
    )
    assert {r["patient"] for r in m} == {"01", "02", "03"}
    assert all("event_auprc" in r for r in em)
    assert not est.__sklearn_is_fitted__()  # cross_val_predict clones -> no leakage


def test_lopocv_does_not_fit_the_passed_estimator():
    # cross_val_predict clones per fold; the shared estimator must stay unfitted.
    # If it were fit once on all data, that would leak the held-out patient.
    X, y, pids, times, seiz = _toy_cohort()
    est = build_logistic()
    leave_one_patient_out_cv(
        X, y, pids, est,
        start_times=times, seizure_intervals_by_patient=seiz, hop_sec=0.5,
    )
    assert not hasattr(est.named_steps["lr"], "coef_")
