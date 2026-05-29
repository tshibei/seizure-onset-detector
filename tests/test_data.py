from pathlib import Path

import numpy as np
import pytest

from seizure_onset_detector.data import (
    label_windows,
    list_patients,
    list_recordings,
    load_info,
    load_recording,
    normalize_recording,
    recording_start_seconds,
    window_signal,
)


def test_normalize_recording_centers_and_scales_per_channel():
    rng = np.random.default_rng(0)
    # Two channels with very different offsets/scales.
    signal = np.stack([
        rng.normal(100.0, 5.0, size=4000),
        rng.normal(-20.0, 50.0, size=4000),
    ])
    out = normalize_recording(signal)

    assert out.shape == signal.shape
    # Per-channel median -> ~0 after centering.
    assert np.allclose(np.median(out, axis=1), 0.0, atol=1e-9)
    # Per-channel MAD -> ~1/1.4826 (since we divide by scale*MAD).
    mad_out = np.median(np.abs(out - np.median(out, axis=1, keepdims=True)), axis=1)
    assert np.allclose(mad_out, 1.0 / 1.4826, rtol=1e-6)


def test_normalize_recording_matches_manual_formula():
    signal = np.array([[1.0, 2.0, 3.0, 100.0]])  # one channel, with an outlier
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    expected = (signal - median) / (1.4826 * mad)
    assert np.allclose(normalize_recording(signal), expected)


def test_normalize_recording_flat_channel_is_zero():
    signal = np.array([[7.0, 7.0, 7.0, 7.0]])  # MAD = 0
    out = normalize_recording(signal)
    assert np.allclose(out, 0.0)  # centered, no division by zero
    assert np.isfinite(out).all()

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
    offset = recording_start_seconds("37h")

    windows, start_times = window_signal(
        signal, fs, window_sec=1.0, overlap=0.5, start_offset_sec=offset
    )

    assert len(windows) == 19
    assert windows[0].shape == (8, fs)
    assert start_times[0] == 36 * 3600  # hour offset applied

def test_label_windows_marks_seizure_overlap():
    """A window inside a seizure interval gets label 1."""
    start_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    seizure_intervals = [(2.5, 3.5)]

    labels = label_windows(start_times, seizure_intervals, window_sec=1.0)

    assert list(labels) == [0, 0, 1, 1, 0]