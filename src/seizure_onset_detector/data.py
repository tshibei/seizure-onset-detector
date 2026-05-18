"""Data loading and preprocessing for iEEG recordings."""
import glob
import json
import os
import re

import numpy as np
import scipy.io


def list_patients(data_dir: str):
    """List patients with at least one recording file in data_dir."""
    files = glob.glob(f"{data_dir}/ID*_*h.mat")
    patients = set()
    for f in files:
        match = re.search(r"ID(\d+)_\d+h\.mat$", os.path.basename(f))
        if match:
            patients.add(match.group(1))
    return sorted(patients, key=int)

def list_recordings(data_dir: str, patient_id: str):
    """List all recordings for a given patient."""
    files = glob.glob(f"{data_dir}/ID{patient_id}_*.mat")
    recordings = []
    for f in files:
        match = re.search(rf"ID{patient_id}_(\d+)h\.mat$", os.path.basename(f))
        if match:
            recordings.append(f"{match.group(1)}h")
    return sorted(recordings, key=lambda r: int(r[:-1]))

def load_info(data_dir: str, patient_id: str):
    """Load info file for a patient. Returns (fs, seizure_intervals)."""
    info_file = f"{data_dir}/ID{patient_id}_info.mat"
    loaded_info = scipy.io.loadmat(info_file)

    fs = float(np.asarray(loaded_info["fs"]).flatten()[0])
    seizure_begin = np.asarray(loaded_info["seizure_begin"]).flatten()
    seizure_end = np.asarray(loaded_info["seizure_end"]).flatten()
    seizure_intervals = list(zip(seizure_begin, seizure_end, strict=True))
    return fs, seizure_intervals

def load_recording(data_dir: str, patient_id: str, recording_id: str):
    """Load an iEEG recording. Returns array of shape (n_channels, n_samples)."""
    file = f"{data_dir}/ID{patient_id}_{recording_id}.mat"
    return scipy.io.loadmat(file)["EEG"]

def load_cohort(cohort_file="scripts/download_manifest.json"):
    """Load the manifest. Keys are patient IDs (no 'ID' prefix), values are recording-hour ints."""
    with open(cohort_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key.removeprefix("ID"): hours for key, hours in raw.items()}

def window_signal(signal, fs, window_sec=1.0, overlap=0.5, start_offset_sec=0.0):
    """Split signal into overlapping windows; start_times are offset by start_offset_sec."""
    n_samples = signal.shape[1]
    window_size = int(window_sec * fs)
    step_size = int(window_size * (1 - overlap))
    windows = []
    start_times = []
    for start in range(0, n_samples - window_size + 1, step_size):
        windows.append(signal[:, start:start + window_size])
        start_times.append(start / fs + start_offset_sec)
    return np.stack(windows), np.array(start_times)

def label_windows(start_times, seizure_intervals, window_sec):
    """Binary labels per window: 1 if window overlaps any seizure interval."""
    labels = np.zeros(len(start_times), dtype=np.int8)
    for i, start in enumerate(start_times):
        end = start + window_sec
        for sz_start, sz_end in seizure_intervals:
            if start < sz_end and end > sz_start:
                labels[i] = 1
    return labels
    
def recording_start_seconds(recording: str) -> int:
    """Seconds from start-of-monitoring to start of this recording file."""
    match = re.match(r"(\d+)h", recording)
    if not match:
        raise ValueError(f"Bad recording name: {recording!r}")
    return (int(match.group(1)) - 1) * 3600
    
if __name__ == "__main__":
    data_dir = "datasets/swec-ethz"
    patients = list_patients(data_dir)
    print(f"Found {len(patients)} patients: {patients[:5]}{'...' if len(patients) > 5 else ''}")
    
    if not patients:
        print("No patients found. Check data_dir.")
        exit(1)
    
    patient = patients[0]
    fs, seizure_intervals = load_info(data_dir, patient)
    recordings = list_recordings(data_dir, patient)
    print(f"Patient {patient}: fs={fs}, "
          f"{len(seizure_intervals)} seizures, {len(recordings)} recordings")

    recording = recordings[0]
    signal = load_recording(data_dir, patient, recording)
    print(f"  {recording}: signal shape {signal.shape}")

    offset = recording_start_seconds(recording)
    windows, start_times = window_signal(signal, fs, start_offset_sec=offset)
    labels = label_windows(start_times, seizure_intervals, window_sec=1.0)
    print(f"  {len(windows)} windows, {sum(labels)} ictal ({100*sum(labels)/len(labels):.2f}%)")