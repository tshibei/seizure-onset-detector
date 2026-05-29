import argparse
from pathlib import Path

import pandas as pd

from seizure_onset_detector.cache import HOP_SEC, load_cohort_features
from seizure_onset_detector.data import list_patients
from seizure_onset_detector.evaluate import leave_one_patient_out_cv
from seizure_onset_detector.models import build_logistic, build_random_forest, build_xgboost

DATA_DIR = "datasets/swec-ethz"
RESULTS_PATH = Path("results/baseline_metrics.csv")
CURVES_PATH = Path("results/baseline_curves.csv")
EVENT_RESULTS_PATH = Path("results/event_metrics.csv")
EVENT_CURVES_PATH = Path("results/event_curves.csv")
CSV_COLUMNS = [
    "model", "patient",
    "sensitivity", "precision", "f1", "fpr_per_hour", "auprc", "auprc_by_chance",
    "n_test", "n_test_pos", "skipped",
]
CURVE_COLUMNS = [
    "model", "patient", "threshold", "fp_per_hour", "sensitivity", "precision",
]
EVENT_CSV_COLUMNS = [
    "model", "patient", "event_sensitivity", "precision", "f1", "fa_per_hour", "event_auprc",
    "median_latency", "mean_latency",
    "n_seizures", "n_detected", "n_false_alarms", "threshold", "skipped",
]
EVENT_CURVE_COLUMNS = [
    "model", "patient", "threshold", "fa_per_hour", "event_sensitivity", "precision", "f1",
    "median_latency", "mean_latency", "n_seizures", "n_detected", "n_false_alarms",
]
ESTIMATORS = {
    "logistic": build_logistic(),
    "random_forest": build_random_forest(),
    "xgboost": build_xgboost(),
}


def main(patients=None, rebuild=False):
    patient_ids = patients or list_patients(data_dir=DATA_DIR)
    print(f"Using {len(patient_ids)} patients: {patient_ids[:5]}{'...' if len(patient_ids) > 5 else ''}")

    X, y, window_patient_ids, window_start_times, seizure_intervals_by_patient = (
        load_cohort_features(DATA_DIR, patient_ids, rebuild=rebuild)
    )

    metric_rows, curve_rows = [], []
    event_metric_rows, event_curve_rows = [], []
    for model_name, estimator in ESTIMATORS.items():
        results, curves, event_results, event_curves = leave_one_patient_out_cv(
            X, y, window_patient_ids, estimator,
            start_times=window_start_times,
            seizure_intervals_by_patient=seizure_intervals_by_patient,
            hop_sec=HOP_SEC,
        )
        print(f"Results for {model_name}:")
        for metrics in results:
            print(metrics)
            metric_rows.append({"model": model_name, **metrics})
        for c in curves:
            curve_rows.append({"model": model_name, **c})
        for em in event_results:
            print(em)
            event_metric_rows.append({"model": model_name, **em})
        for ec in event_curves:
            event_curve_rows.append({"model": model_name, **ec})

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = (
        pd.DataFrame(metric_rows)
        .reindex(columns=CSV_COLUMNS)
        .sort_values(["model", "patient"])
        .reset_index(drop=True)
    )
    metrics_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved {len(metrics_df)} rows to {RESULTS_PATH}")

    curves_df = (
        pd.DataFrame(curve_rows)
        .reindex(columns=CURVE_COLUMNS)
        .sort_values(["model", "patient", "threshold"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    curves_df.to_csv(CURVES_PATH, index=False)
    print(f"Saved {len(curves_df)} rows to {CURVES_PATH}")

    event_metrics_df = (
        pd.DataFrame(event_metric_rows)
        .reindex(columns=EVENT_CSV_COLUMNS)
        .sort_values(["model", "patient"])
        .reset_index(drop=True)
    )
    event_metrics_df.to_csv(EVENT_RESULTS_PATH, index=False)
    print(f"Saved {len(event_metrics_df)} rows to {EVENT_RESULTS_PATH}")

    event_curves_df = (
        pd.DataFrame(event_curve_rows)
        .reindex(columns=EVENT_CURVE_COLUMNS)
        .sort_values(["model", "patient", "threshold"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    event_curves_df.to_csv(EVENT_CURVES_PATH, index=False)
    print(f"Saved {len(event_curves_df)} rows to {EVENT_CURVES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baselines with LOPOCV, reusing cached per-patient features."
    )
    parser.add_argument(
        "--patients", nargs="+", metavar="ID",
        help="Subset of patient IDs to use (e.g. --patients 03 04). Default: all.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-extract features from raw recordings, ignoring any cache.",
    )
    args = parser.parse_args()
    main(patients=args.patients, rebuild=args.rebuild)
