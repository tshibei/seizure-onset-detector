from pathlib import Path

import numpy as np
import pandas as pd

from seizure_onset_detector.data import (
    label_windows,
    list_patients,
    list_recordings,
    load_info,
    load_recording,
    recording_start_seconds,
    window_signal,
)
from seizure_onset_detector.evaluate import leave_one_patient_out_cv
from seizure_onset_detector.features import extract_features_batch
from seizure_onset_detector.models import train_logistic, train_random_forest

DATA_DIR = "datasets/swec-ethz"
RESULTS_PATH = Path("results/baseline_metrics.csv")
CSV_COLUMNS = [
    "model", "patient",
    "sensitivity", "fpr_per_hour", "auprc", "auprc_by_chance",
    "n_test", "n_test_pos", "skipped",
]
TRAINERS = {
    "logistic": train_logistic,
    "random_forest": train_random_forest,
}


def main():
    patient_ids = list_patients(data_dir=DATA_DIR)
    print(f"Found {len(patient_ids)} patients: {patient_ids[:5]}{'...' if len(patient_ids) > 5 else ''}")

    X_parts, y_parts, pid_parts = [], [], []
    for patient_id in patient_ids:
        print(f"Loading data for patient {patient_id}...")
        fs, seizure_intervals = load_info(DATA_DIR, patient_id)
        recordings = list_recordings(DATA_DIR, patient_id)
        for recording in recordings:
            signal = load_recording(DATA_DIR, patient_id, recording)
            offset = recording_start_seconds(recording)
            windows, start_times = window_signal(signal, fs, start_offset_sec=offset)
            labels = label_windows(start_times, seizure_intervals, window_sec=1.0)

            features = extract_features_batch(windows, fs)
            X_parts.append(features)
            y_parts.append(np.asarray(labels))
            pid_parts.append(np.full(len(labels), patient_id))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    window_patient_ids = np.concatenate(pid_parts, axis=0)

    all_rows = []
    for model_name, trainer in TRAINERS.items():
        results = leave_one_patient_out_cv(X, y, window_patient_ids, trainer)
        print(f"Results for {model_name}:")
        for metrics in results:
            print(metrics)
            all_rows.append({"model": model_name, **metrics})

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = (
        pd.DataFrame(all_rows)
        .reindex(columns=CSV_COLUMNS)
        .sort_values(["model", "patient"])
        .reset_index(drop=True)
    )
    df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved {len(df)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
