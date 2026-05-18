import pytest
import numpy as np
from pathlib import Path
from seizure_onset_detector.data import (
    list_patients,
    list_recordings,
    load_info,
    load_recording,
    window_signal,
    label_windows
)

DATA_DIR = Path("datasets/swec-ethz")
@pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not available")
def test_load_recording_returns_expected_tuple():
    patients = list_patients(DATA_DIR)
    assert len(patients) > 0, "No patients found in the dataset directory."

    recordings = list_recordings(DATA_DIR, patients[0])
    assert len(recordings) > 0, f"No recordings found for patient {patients[0]}."

    fs, seizure_intervals = load_info(DATA_DIR, patients[0])
    assert fs > 0, "Sampling frequency should be a positive number."

    signal = load_recording(DATA_DIR, patients[0], recordings[0])
    assert signal.ndim == 2, "Expected signal to be a 2D array (n_channels, n_samples)."
    assert signal.shape[0] > 0, "Expected at least one channel in the signal."
    assert signal.shape[1] > 0, "Expected at least one sample in the signal."

def test_window_signal_basic():
    """Shape, count, and hour offset in one go."""
    fs = 256
    signal = np.random.randn(8, fs * 10)
    
    windows, start_times = window_signal("37h", signal, fs, window_sec=1.0, overlap=0.5)
    
    assert len(windows) == 19
    assert windows[0].shape == (8, fs)
    assert start_times[0] == 36 * 3600  # hour offset applied

def test_label_windows_marks_seizure_overlap():
    """A window inside a seizure interval gets label 1."""
    fs = 256
    windows = [np.zeros((4, fs))] * 5
    start_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    seizure_intervals = [(2.5, 3.5)]
    
    labels = label_windows(windows, start_times, seizure_intervals, fs)
    
    assert labels == [0, 0, 1, 1, 0]