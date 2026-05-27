"""Baseline classifiers for seizure onset detection."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic(X, y):
    """Train a logistic regression classifier with feature standardization."""
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    clf.fit(X, y)
    return clf


def train_random_forest(X, y):
    """Train a random forest classifier."""
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X, y)
    return clf
