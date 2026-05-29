"""Baseline classifiers for seizure onset detection.

Builders return *unfitted* estimators so leave_one_patient_out_cv can hand them
to cross_val_predict, which clones and refits per fold -- the scaler is therefore
fit on each fold's training patients only and applied to the held-out patient,
with no leakage.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def build_logistic():
    """Unfitted logistic-regression pipeline with RobustScaler normalization.

    RobustScaler (median-centered, IQR-scaled) is used instead of StandardScaler
    because iEEG band-power/line-length features have heavy outliers around
    seizures that would distort a mean/variance scaler.
    """
    return Pipeline([
        ("scaler", RobustScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def build_random_forest():
    """Unfitted random forest. No scaler: tree splits are invariant to the
    per-feature monotonic transform a scaler applies, so it would be a no-op."""
    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )


def build_xgboost():
    """Unfitted XGBoost gradient-boosted trees with per-fold class balancing.

    scale_pos_weight is set to n_neg / n_pos from each fold's *training* labels at
    fit time -- the analog of class_weight="balanced" used by the other models --
    so the heavy seizure/interictal imbalance is handled per LOPOCV fold without
    leaking the held-out patient. No feature scaler (tree-based).

    XGBoost is imported lazily because its native library needs an OpenMP runtime
    (libomp on macOS, libgomp on Linux). The balancing wrapper is a locally-defined
    subclass (not picklable), so leave_one_patient_out_cv must keep cross_val_predict
    sequential (its default); parallelising folds would need a module-level wrapper.
    """
    from xgboost import XGBClassifier

    class _BalancedXGBClassifier(XGBClassifier):
        def fit(self, X, y, **kwargs):
            y = np.asarray(y)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            self.scale_pos_weight = n_neg / n_pos if n_pos else 1.0
            return super().fit(X, y, **kwargs)

    return _BalancedXGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
