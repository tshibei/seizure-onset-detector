import numpy as np

from seizure_onset_detector.evaluate import (
    binarize_to_events,
    compute_event_auprc,
    compute_event_metrics,
    compute_metrics,
    match_events_overlap,
)

HOP = 0.5
WINDOW = 1.0


def _timeline(end_sec, positive_ranges):
    """Build (start_times, scores) at HOP spacing; score=1 inside any positive range."""
    start_times = np.arange(0.0, end_sec + HOP / 2, HOP)
    scores = np.zeros_like(start_times)
    for lo, hi in positive_ranges:
        scores[(start_times >= lo) & (start_times <= hi)] = 1.0
    return start_times, scores


def test_binarize_keeps_sustained_drops_flicker():
    # 5 s sustained run (kept), an isolated single-window blip (dropped).
    start_times, scores = _timeline(60.0, positive_ranges=[(10.0, 14.0), (50.0, 50.0)])
    events = binarize_to_events(
        start_times, scores, threshold=0.5, window_sec=WINDOW, min_duration_sec=3.0,
    )
    assert len(events) == 1
    onset, offset = events[0]
    assert onset == 10.0 and offset == 15.0  # last positive window 14.0 + WINDOW


def test_binarize_overlap_only_no_window_bridging():
    # Two 5 s runs with a 5 s gap: NOT bridged at the window level (each its own raw
    # alarm); event_merge_gap_sec bridges them at the alarm level instead.
    start_times, scores = _timeline(40.0, positive_ranges=[(10.0, 14.0), (20.0, 24.0)])
    common = dict(window_sec=WINDOW, min_duration_sec=3.0)
    assert len(binarize_to_events(start_times, scores, 0.5, **common, event_merge_gap_sec=0.0)) == 2
    merged = binarize_to_events(start_times, scores, 0.5, **common, event_merge_gap_sec=10.0)
    assert merged == [(10.0, 25.0)]   # alarm-level merge bridges the 5 s gap


def test_binarize_splits_alarms_over_max_duration():
    # One 12-min sustained alarm (window starts every 0.5 s, 0..720 s).
    start_times, scores = _timeline(740.0, positive_ranges=[(0.0, 720.0)])
    common = dict(window_sec=WINDOW, min_duration_sec=3.0, event_merge_gap_sec=0.0)
    whole = binarize_to_events(start_times, scores, 0.5, **common, max_event_duration_sec=0.0)
    assert len(whole) == 1 and whole[0] == (0.0, 721.0)  # 720 + WINDOW, unsplit
    chunks = binarize_to_events(start_times, scores, 0.5, **common, max_event_duration_sec=300.0)
    assert [(round(s, 1), round(e, 1)) for s, e in chunks] == [(0.0, 300.0), (300.0, 600.0), (600.0, 721.0)]
    assert all(e - s <= 300.0 for s, e in chunks)  # remainder (121 s) kept as its own chunk


def test_binarize_merges_alarms_within_event_gap():
    # Two sustained alarms 25 s apart merge under event_merge_gap_sec; a far one stays.
    start_times, scores = _timeline(250.0, positive_ranges=[(10.0, 14.0), (40.0, 44.0), (200.0, 204.0)])
    common = dict(window_sec=WINDOW, min_duration_sec=3.0)
    assert len(binarize_to_events(start_times, scores, 0.5, **common, event_merge_gap_sec=0.0)) == 3
    merged = binarize_to_events(start_times, scores, 0.5, **common, event_merge_gap_sec=90.0)
    assert len(merged) == 2            # [10,15] and [40,45] merge; [200,205] stays separate
    assert merged[0] == (10.0, 45.0)   # spans first onset to second offset


def test_window_precision_and_f1():
    # Asymmetric (precision != recall != f1) so a precision/recall swap is caught.
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1])  # tp=3, fp=2, fn=0
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55])
    m = compute_metrics(y_true, y_pred, y_scores)
    assert m["sensitivity"] == 1.0    # 3 / (3 + 0)
    assert m["precision"] == 0.6      # 3 / (3 + 2)
    assert m["f1"] == 0.75            # 2*3 / (2*3 + 2 + 0)


def test_event_precision_and_f1():
    # Two seizures both detected + one interictal false alarm: precision != recall != f1.
    start_times, scores = _timeline(
        1000.0, positive_ranges=[(110.0, 114.0), (510.0, 514.0), (900.0, 904.0)]
    )
    seizures = [(100.0, 130.0), (500.0, 530.0)]
    m = compute_event_metrics(
        start_times, scores, seizures, threshold=0.5,
        window_sec=WINDOW, hop_sec=HOP, min_duration_sec=3.0,
    )
    assert m["n_detected"] == 2 and m["n_false_alarms"] == 1 and m["n_seizures"] == 2
    assert m["event_sensitivity"] == 1.0      # 2 of 2 seizures
    assert np.isclose(m["precision"], 2 / 3)  # 2 detections / (2 + 1 false alarm)
    assert m["f1"] == 0.8                      # 2*2 / (2*2 + 1 + 0)


def test_match_detects_on_any_overlap():
    seizures = [(100.0, 130.0)]
    pred = [(110.0, 116.0)]  # overlaps the seizure
    m = match_events_overlap(pred, seizures)
    assert m["n_detected"] == 1
    assert m["n_false_alarms"] == 0
    assert m["latencies"] == [10.0]  # earliest overlapping onset - sz_start


def test_match_overlap_anywhere_in_seizure_is_a_detection():
    seizures = [(100.0, 200.0)]
    pred = [(140.0, 150.0)]  # overlaps deep inside the seizure -> detection under any-overlap
    m = match_events_overlap(pred, seizures)
    assert m["n_detected"] == 1
    assert m["n_false_alarms"] == 0
    assert m["latencies"] == [40.0]


def test_match_interictal_alarm_is_false_alarm():
    seizures = [(100.0, 130.0)]
    pred = [(300.0, 305.0)]  # fires when no seizure is occurring
    m = match_events_overlap(pred, seizures)
    assert m["n_detected"] == 0
    assert m["n_false_alarms"] == 1


def test_match_missed_seizure_has_no_overlapping_alarm():
    seizures = [(100.0, 130.0), (500.0, 530.0)]
    pred = [(110.0, 116.0)]  # overlaps only the first seizure
    m = match_events_overlap(pred, seizures)
    assert m["n_detected"] == 1  # second seizure has no overlapping alarm -> miss
    assert m["n_seizures"] == 2


def test_pre_ictal_alarm_within_tolerance_is_detection():
    seizures = [(100.0, 130.0)]
    within = match_events_overlap([(80.0, 85.0)], seizures)   # 20 s before onset, inside 30 s tol
    assert within["n_detected"] == 1 and within["n_false_alarms"] == 0
    assert within["latencies"] == [-20.0]                     # fired before onset
    outside = match_events_overlap([(60.0, 64.0)], seizures)  # ends before the 30 s tol -> FA
    assert outside["n_detected"] == 0 and outside["n_false_alarms"] == 1


def test_post_ictal_alarm_within_tolerance_is_detection():
    seizures = [(100.0, 130.0)]                                  # offset 130, post-ictal tol -> 190
    inside = match_events_overlap([(150.0, 160.0)], seizures)
    assert inside["n_detected"] == 1 and inside["n_false_alarms"] == 0
    outside = match_events_overlap([(195.0, 200.0)], seizures)   # starts after 190 -> FA
    assert outside["n_detected"] == 0 and outside["n_false_alarms"] == 1


def test_match_merges_seizures_within_gap():
    seizures = [(100.0, 130.0), (200.0, 210.0)]   # gap 70 s < 90 -> merged into one event
    m = match_events_overlap([(205.0, 209.0)], seizures)
    assert m["n_seizures"] == 1
    assert m["n_detected"] == 1


def test_compute_event_metrics_end_to_end():
    # One detected seizure, one far interictal false alarm, one sub-duration blip.
    start_times, scores = _timeline(
        600.0, positive_ranges=[(110.0, 114.0), (500.0, 504.0), (200.0, 200.0)]
    )
    seizures = [(100.0, 130.0)]
    m = compute_event_metrics(
        start_times, scores, seizures, threshold=0.5,
        window_sec=WINDOW, hop_sec=HOP,
        min_duration_sec=3.0,
    )
    assert m["n_seizures"] == 1
    assert m["n_detected"] == 1  # alarm [110,115] overlaps the seizure tolerance window
    assert m["event_sensitivity"] == 1.0
    assert m["n_false_alarms"] == 1  # [500,505] is far from the seizure; blip at 200 filtered
    assert m["median_latency"] == 10.0

    n_windows = len(scores)
    expected_hours = n_windows * HOP / 3600
    assert np.isclose(m["fa_per_hour"], 1 / expected_hours)


def test_event_auprc_max_duration_split_breaks_all_fire_degeneracy():
    # The all-fire point (threshold 0 -> one giant alarm spanning the recording) would,
    # without splitting, overlap the seizure with zero false alarms (precision 1, a
    # degenerate (1, 1) PR point). With max_event_duration_sec the giant alarm is cut
    # into <=5-min chunks: the chunk over the seizure detects it, the interictal chunks
    # become false alarms -> precision 0.5 at full recall, so AUPRC is no longer ~1.0.
    start_times = np.arange(0.0, 400.0 + HOP / 2, HOP)
    scores = np.zeros_like(start_times)
    scores[(start_times >= 110) & (start_times <= 114)] = 0.9  # overlaps the seizure
    scores[(start_times >= 300) & (start_times <= 304)] = 0.9  # interictal false alarm
    seizures = [(100.0, 130.0)]

    ap = compute_event_auprc(
        start_times, scores, seizures,
        window_sec=WINDOW, min_duration_sec=3.0, max_event_duration_sec=300.0,
    )
    assert np.isclose(ap, 0.75)


def test_event_auprc_nan_without_seizures():
    start_times = np.arange(0.0, 100.0, HOP)
    scores = np.zeros_like(start_times)
    assert np.isnan(compute_event_auprc(start_times, scores, seizure_intervals=[]))


def test_compute_event_metrics_no_seizures_returns_nan_sensitivity():
    start_times, scores = _timeline(100.0, positive_ranges=[(10.0, 14.0)])
    m = compute_event_metrics(
        start_times, scores, seizure_intervals=[], threshold=0.5,
        window_sec=WINDOW, hop_sec=HOP,
    )
    assert m["n_seizures"] == 0
    assert np.isnan(m["event_sensitivity"])
    assert m["n_false_alarms"] == 1  # the alarm has no seizure to overlap
