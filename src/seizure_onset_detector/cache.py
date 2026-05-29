"""Per-patient feature cache.

Feature extraction (iEEG file I/O + notch filter + DSP) is the only slow part of
the pipeline; the resulting matrix is tiny. Caching it per patient lets the
train/eval loop skip extraction entirely, re-extract a single patient when its
recordings or the feature code change, and load just a few patients for fast
experiments. Caches live under datasets/cache/ (already gitignored).
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from seizure_onset_detector import features as _features
from seizure_onset_detector.data import (
    label_windows,
    list_recordings,
    load_info,
    load_recording,
    normalize_recording,
    notch_filter,
    recording_start_seconds,
    window_signal,
)
from seizure_onset_detector.features import extract_features_batch, feature_names

CACHE_DIR = Path("datasets/cache")
WINDOW_SEC = 1.0
OVERLAP = 0.5
NOTCH_FREQ = 50.0
NORMALIZE = True       # per-channel MAD normalization of each recording
MAD_SCALE = 1.4826     # MAD -> std consistency factor used by normalize_recording
HOP_SEC = WINDOW_SEC * (1 - OVERLAP)  # 0.5 s between consecutive window starts


def _feature_code_hash():
    """SHA-256 of the features module source, so editing feature *computation*
    (not just the feature names) invalidates caches. Any edit to features.py --
    including comments and whitespace -- changes this hash."""
    return hashlib.sha256(inspect.getsource(_features).encode("utf-8")).hexdigest()


def _extraction_params():
    """Stamp identifying how features were extracted; a mismatch invalidates a cache."""
    return {
        "window_sec": WINDOW_SEC,
        "overlap": OVERLAP,
        "notch_freq": NOTCH_FREQ,
        "normalize": NORMALIZE,
        "mad_scale": MAD_SCALE if NORMALIZE else None,
        "feature_names": feature_names(),
        "feature_code_sha256": _feature_code_hash(),
    }


def cache_path(patient_id, cache_dir=CACHE_DIR):
    return Path(cache_dir) / f"features_ID{patient_id}.npz"


def build_patient_features(data_dir, patient_id):
    """The slow path: load, notch-filter, window, and extract features for one patient.

    Returns a dict with float32 X, int8 y, float64 start_times, and a (n, 2)
    seizure_intervals array.
    """
    fs, seizure_intervals = load_info(data_dir, patient_id)
    X_parts, y_parts, t_parts = [], [], []
    for recording in list_recordings(data_dir, patient_id):
        signal = load_recording(data_dir, patient_id, recording)
        signal = notch_filter(signal, fs, freq=NOTCH_FREQ)
        if NORMALIZE:
            signal = normalize_recording(signal, scale=MAD_SCALE)
        offset = recording_start_seconds(recording)
        windows, start_times = window_signal(
            signal, fs, window_sec=WINDOW_SEC, overlap=OVERLAP, start_offset_sec=offset
        )
        labels = label_windows(start_times, seizure_intervals, window_sec=WINDOW_SEC)
        X_parts.append(extract_features_batch(windows, fs))
        y_parts.append(np.asarray(labels))
        t_parts.append(start_times)
    return {
        "X": np.concatenate(X_parts).astype(np.float32),
        "y": np.concatenate(y_parts).astype(np.int8),
        "start_times": np.concatenate(t_parts).astype(np.float64),
        "seizure_intervals": np.asarray(
            [(float(s), float(e)) for s, e in seizure_intervals], dtype=np.float64
        ).reshape(-1, 2),
    }


def save_patient_cache(patient_id, data, cache_dir=CACHE_DIR):
    path = cache_path(patient_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=data["X"],
        y=data["y"],
        start_times=data["start_times"],
        seizure_intervals=data["seizure_intervals"],
        params=np.array(json.dumps(_extraction_params())),
    )
    return path


def load_patient_cache(patient_id, cache_dir=CACHE_DIR):
    """Return cached arrays, or None if the cache is absent or stale.

    A cache is stale when it was written with different extraction params (window
    size, overlap, notch frequency, or the set/order of features).
    """
    path = cache_path(patient_id, cache_dir)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        if json.loads(str(npz["params"])) != _extraction_params():
            print(f"  cache for patient {patient_id} is stale (extraction params changed)")
            return None
        return {
            "X": npz["X"],
            "y": npz["y"],
            "start_times": npz["start_times"],
            "seizure_intervals": [tuple(row) for row in npz["seizure_intervals"]],
        }


def build_or_load_patient(data_dir, patient_id, rebuild=False, cache_dir=CACHE_DIR):
    """Load a patient's cached features, building (and caching) them on a miss."""
    if not rebuild:
        cached = load_patient_cache(patient_id, cache_dir)
        if cached is not None:
            print(f"  loaded patient {patient_id} from cache")
            return cached
    print(f"  extracting features for patient {patient_id}...")
    data = build_patient_features(data_dir, patient_id)
    save_patient_cache(patient_id, data, cache_dir)
    return {
        "X": data["X"],
        "y": data["y"],
        "start_times": data["start_times"],
        "seizure_intervals": [tuple(row) for row in data["seizure_intervals"]],
    }


def load_cohort_features(data_dir, patient_ids, rebuild=False, cache_dir=CACHE_DIR):
    """Assemble the cohort feature matrix from per-patient caches.

    Returns (X, y, patient_ids, start_times, seizure_intervals_by_patient), ready
    to hand to leave_one_patient_out_cv.
    """
    X_parts, y_parts, pid_parts, t_parts = [], [], [], []
    seizure_intervals_by_patient = {}
    for pid in patient_ids:
        d = build_or_load_patient(data_dir, pid, rebuild=rebuild, cache_dir=cache_dir)
        X_parts.append(d["X"])
        y_parts.append(d["y"])
        t_parts.append(d["start_times"])
        pid_parts.append(np.full(len(d["y"]), pid))
        seizure_intervals_by_patient[pid] = d["seizure_intervals"]
    return (
        np.concatenate(X_parts),
        np.concatenate(y_parts),
        np.concatenate(pid_parts),
        np.concatenate(t_parts),
        seizure_intervals_by_patient,
    )
