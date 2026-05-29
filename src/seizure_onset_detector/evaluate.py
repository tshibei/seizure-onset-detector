"""Evaluation metrics for seizure onset detection."""
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict


def compute_metrics(y_true, y_pred, y_scores, window_sec=1.0):
    """Compute sensitivity, precision, F1, FPR/hour, and AUPRC.

    Sensitivity/precision/F1 are NaN when their denominator is empty (e.g. no
    positive predictions). AUPRC is NaN if y_true is single-class (curve is
    undefined). Forcing labels=[0, 1] on confusion_matrix keeps the 2x2 shape
    even when a fold's predictions or labels collapse to one class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_test = len(y_true)
    n_test_pos = int((y_true == 1).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    total_hours = (n_test * window_sec) / 3600
    fpr_per_hour = fp / total_hours if total_hours > 0 else float("nan")

    auprc_by_chance = n_test_pos / n_test if n_test > 0 else float("nan")

    if 0 < n_test_pos < n_test:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = auc(pr_recall, pr_precision)
    else:
        auprc = float("nan")

    return {
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": f1,
        "fpr_per_hour": fpr_per_hour,
        "auprc": auprc,
        "auprc_by_chance": auprc_by_chance,
        "n_test": n_test,
        "n_test_pos": n_test_pos,
    }


def compute_curve(y_true, y_scores, window_sec=1.0):
    """Threshold sweep from roc_curve, expressed in (FP/h, sensitivity, precision).

    Returns a list of dicts (one per distinct operating point). Empty list if
    y_true is single-class, since the curve is undefined there.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return []

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    total_hours = (n * window_sec) / 3600
    fp_per_hour = fpr * n_neg / total_hours
    tp = tpr * n_pos
    fp = fpr * n_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)

    return [
        {
            "threshold": float(t),
            "fp_per_hour": float(fph),
            "sensitivity": float(s),
            "precision": float(p),
        }
        for t, fph, s, p in zip(thresholds, fp_per_hour, tpr, precision, strict=True)
    ]


def _merge_intervals(intervals, gap_sec):
    """Merge (start, end) intervals whose separation (next start - prev end) is < gap_sec.

    Shared by the true-seizure merge and the alarm-event merge. Returns a new
    onset-sorted list of (start, end) tuples.
    """
    ordered = sorted((float(s), float(e)) for s, e in intervals)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s - merged[-1][1] < gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _split_long_events(events, max_dur_sec):
    """Cut alarms longer than max_dur_sec into consecutive chunks of <= max_dur_sec.

    Chunks are measured from each alarm's own onset; the final chunk keeps the
    (shorter) remainder. No-op when max_dur_sec <= 0.
    """
    if max_dur_sec <= 0:
        return events
    out = []
    for s, e in events:
        t = s
        while t < e:
            out.append((t, min(t + max_dur_sec, e)))
            t += max_dur_sec
    return out


def binarize_to_events(
    start_times,
    scores,
    threshold,
    window_sec=1.0,
    min_duration_sec=5.0,
    event_merge_gap_sec=0.0,
    max_event_duration_sec=0.0,
):
    """Sustained-duration debounce: per-window scores -> discrete alarm intervals.

    Windows scoring >= threshold contribute their [start, start+window_sec] span.
    Touching/overlapping spans form one raw alarm, and only alarms lasting
    >= min_duration_sec are kept (this suppresses isolated flicker). Surviving
    alarms separated by < event_merge_gap_sec are then merged into a single
    detection, and finally any alarm longer than max_event_duration_sec is split
    into consecutive chunks of that length. Returns a list of (onset, offset)
    tuples in start_times units.
    """
    start_times = np.asarray(start_times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(start_times, kind="stable")
    st = start_times[order][scores[order] >= threshold]  # sorted starts of positive windows
    if st.size == 0:
        return []

    ends = st + window_sec  # monotonic, since st is sorted and window_sec is constant
    # A new alarm begins wherever a positive window does not touch/overlap the previous.
    new_group = np.concatenate(([True], (st[1:] - ends[:-1]) > 0))
    starts = np.flatnonzero(new_group)
    last = np.append(starts[1:] - 1, st.size - 1)  # last window index in each group
    onsets, offsets = st[starts], ends[last]

    keep = (offsets - onsets) >= min_duration_sec
    events = list(zip(onsets[keep], offsets[keep], strict=True))

    if event_merge_gap_sec > 0:
        events = _merge_intervals(events, event_merge_gap_sec)
    if max_event_duration_sec > 0:
        events = _split_long_events(events, max_event_duration_sec)
    return events


def match_events_overlap(
    pred_events,
    seizure_intervals,
    pre_ictal_sec=30.0,
    post_ictal_sec=60.0,
    merge_seizure_gap_sec=90.0,
):
    """Match alarms to seizures by overlap within a peri-ictal tolerance.

    True seizures separated by < merge_seizure_gap_sec are merged into one event
    first. Each (merged) seizure's match window is the tolerance-extended interval
    [start - pre_ictal_sec, end + post_ictal_sec]. A seizure is *detected* if some
    alarm overlaps its window (a *miss* otherwise); an alarm is a *false alarm*
    only if it overlaps no seizure window -- i.e. it fires outside every seizure's
    tolerance. Latency is (earliest overlapping alarm onset - seizure start), which
    is negative when the alarm began within the pre-ictal tolerance.
    """
    seizures = _merge_intervals(seizure_intervals, merge_seizure_gap_sec)
    n_detected = 0
    latencies = []

    for sz_start, sz_end in seizures:
        lo, hi = sz_start - pre_ictal_sec, sz_end + post_ictal_sec
        onsets = [p_start for p_start, p_end in pred_events if p_start < hi and p_end > lo]
        if onsets:
            n_detected += 1
            latencies.append(min(onsets) - sz_start)

    false_alarms = sum(
        1
        for p_start, p_end in pred_events
        if not any(
            p_start < sz_end + post_ictal_sec and p_end > sz_start - pre_ictal_sec
            for sz_start, sz_end in seizures
        )
    )

    return {
        "n_seizures": len(seizures),
        "n_detected": int(n_detected),
        "n_false_alarms": int(false_alarms),
        "latencies": latencies,
    }


def compute_event_metrics(
    start_times,
    scores,
    seizure_intervals,
    threshold,
    window_sec=1.0,
    hop_sec=0.5,
    min_duration_sec=5.0,
    event_merge_gap_sec=90.0,
    max_event_duration_sec=300.0,
    pre_ictal_sec=30.0,
    post_ictal_sec=60.0,
):
    """Event-level summary at a single threshold.

    Recording hours are derived as n_windows * hop_sec (each window advances time
    by one hop), which accounts for window overlap unlike a per-window count.
    Alarms within event_merge_gap_sec are merged and alarms longer than
    max_event_duration_sec are split into chunks (see binarize_to_events).
    Detection/false-alarm/miss use overlap matching with a peri-ictal tolerance
    (see match_events_overlap). Precision/F1 are seizure-centric: detected seizures
    are TP, false-alarm events FP, missed seizures FN.
    """
    pred = binarize_to_events(
        start_times, scores, threshold, window_sec, min_duration_sec,
        event_merge_gap_sec, max_event_duration_sec,
    )
    m = match_events_overlap(pred, seizure_intervals, pre_ictal_sec, post_ictal_sec, event_merge_gap_sec)

    n = len(np.asarray(scores))
    total_hours = (n * hop_sec) / 3600 if n > 0 else float("nan")
    n_sz = m["n_seizures"]
    tp, fp = m["n_detected"], m["n_false_alarms"]
    fn = n_sz - tp
    lat = np.asarray(m["latencies"], dtype=float)

    return {
        "threshold": float(threshold),
        "event_sensitivity": tp / n_sz if n_sz > 0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan"),
        "fa_per_hour": fp / total_hours if total_hours and total_hours > 0 else float("nan"),
        "median_latency": float(np.median(lat)) if lat.size else float("nan"),
        "mean_latency": float(np.mean(lat)) if lat.size else float("nan"),
        "n_seizures": n_sz,
        "n_detected": tp,
        "n_false_alarms": fp,
    }


def compute_event_curve(
    start_times,
    scores,
    seizure_intervals,
    window_sec=1.0,
    hop_sec=0.5,
    min_duration_sec=3.0,
    n_thresholds=50,
    event_merge_gap_sec=90.0,
    max_event_duration_sec=300.0,
    pre_ictal_sec=30.0,
    post_ictal_sec=60.0,
):
    """Event-level operating curve: sweep threshold -> (FA/h, event sensitivity, latency).

    Thresholds are the unique scores, sub-sampled by quantile to at most
    n_thresholds points so dense logistic scores don't blow up the sweep.
    Returns a list of compute_event_metrics dicts. Empty if seizures or scores
    are absent (curve undefined).
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or len(seizure_intervals) == 0:
        return []

    uniq = np.unique(scores)
    if uniq.size > n_thresholds:
        thresholds = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, n_thresholds)))
    else:
        thresholds = uniq

    return [
        compute_event_metrics(
            start_times, scores, seizure_intervals, t, window_sec, hop_sec,
            min_duration_sec, event_merge_gap_sec, max_event_duration_sec,
            pre_ictal_sec, post_ictal_sec,
        )
        for t in thresholds
    ]


def compute_event_auprc(
    start_times,
    scores,
    seizure_intervals,
    window_sec=1.0,
    min_duration_sec=3.0,
    n_thresholds=200,
    event_merge_gap_sec=90.0,
    max_event_duration_sec=300.0,
    pre_ictal_sec=30.0,
    post_ictal_sec=60.0,
):
    """Area under the event-level precision-recall curve.

    Sweeps thresholds, and at each computes event recall (= sensitivity) and
    event precision (= detections / (detections + false alarms)) via the same
    debounce + post-onset matching used elsewhere. Integrates the precision
    *envelope* (max precision at each achieved recall) over recall with the
    trapezoid rule -- matching the trapezoidal AUPRC convention used for the
    window-level metric. The envelope keeps the area well-defined even though
    event recall is not monotonic in threshold (debounce can split/merge alarms).

    Each threshold costs a full debounce+match pass, so the sweep is capped at
    n_thresholds quantile points of the unique scores (every unique score is used
    when there are fewer, which is exact for coarse-score models like the forest).
    Returns NaN if the patient has no seizures.
    """
    n_sz = len(seizure_intervals)
    if n_sz == 0:
        return float("nan")
    scores = np.asarray(scores, dtype=float)
    uniq = np.unique(scores)
    if uniq.size > n_thresholds:
        thresholds = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, n_thresholds)))
    else:
        thresholds = uniq

    best_precision = {}  # achieved recall -> max precision seen at that recall
    for t in thresholds:
        pred = binarize_to_events(
            start_times, scores, t, window_sec, min_duration_sec,
            event_merge_gap_sec, max_event_duration_sec,
        )
        m = match_events_overlap(pred, seizure_intervals, pre_ictal_sec, post_ictal_sec, event_merge_gap_sec)
        recall = m["n_detected"] / n_sz
        denom = m["n_detected"] + m["n_false_alarms"]
        precision = m["n_detected"] / denom if denom > 0 else 1.0
        best_precision[recall] = max(best_precision.get(recall, 0.0), precision)

    best_precision.setdefault(0.0, 1.0)  # anchor the curve at recall 0
    recalls = np.array(sorted(best_precision))
    precisions = np.array([best_precision[r] for r in recalls])
    if recalls.size < 2:
        return 0.0  # recall never rises above 0
    return float(auc(recalls, precisions))


def leave_one_patient_out_cv(
    X,
    y,
    patient_ids,
    estimator,
    window_sec=1.0,
    start_times=None,
    seizure_intervals_by_patient=None,
    hop_sec=0.5,
    event_threshold=0.5,
    event_kwargs=None,
):
    """Leave-one-patient-out CV via sklearn LeaveOneGroupOut + cross_val_predict.

    `estimator` is an *unfitted* sklearn estimator (e.g. build_logistic()).
    cross_val_predict clones and refits it on each fold's training patients and
    predicts the held-out patient, so any scaler inside the estimator is fit on
    train only and applied to test -- normalization with no leakage. The returned
    out-of-fold probabilities are then sliced per patient to compute the same
    per-fold metrics as before.

    Returns (metrics, curves, event_metrics, event_curves):
      - metrics: per-fold window-level summary dicts (one per held-out patient)
      - curves: flat list of window-level per-threshold rows tagged with patient
      - event_metrics: per-fold event-level summary dicts (empty unless both
        start_times and seizure_intervals_by_patient are supplied)
      - event_curves: flat list of event-level per-threshold rows tagged with
        patient (empty under the same condition)

    Event scoring needs per-window start_times (aligned to X rows) and the true
    seizure_intervals_by_patient. event_threshold sets the operating point for the
    event-level summary; event_kwargs overrides the debounce/matching parameters.
    """
    patient_ids = np.asarray(patient_ids)
    do_events = start_times is not None and seizure_intervals_by_patient is not None
    if do_events:
        start_times = np.asarray(start_times)
    event_kwargs = dict(event_kwargs or {})

    # Leakage-free out-of-fold positive-class probabilities for every window: each
    # sample is scored by a model trained on all *other* patients.
    print(f"  cross_val_predict over {len(np.unique(patient_ids))} patient folds...", flush=True)
    scores = cross_val_predict(
        estimator, X, y,
        groups=patient_ids,
        cv=LeaveOneGroupOut(),
        method="predict_proba",
        verbose=2,
    )[:, 1]

    metrics_results = []
    curve_rows = []
    event_metrics_results = []
    event_curve_rows = []

    for patient in np.unique(patient_ids):
        mask = patient_ids == patient
        y_test = y[mask]
        y_scores = scores[mask]
        y_pred = (y_scores > 0.5).astype(int)  # matches predict(): argmax breaks 0.5 ties to class 0

        metrics = compute_metrics(y_test, y_pred, y_scores, window_sec=window_sec)
        metrics["patient"] = str(patient)
        metrics_results.append(metrics)

        for row in compute_curve(y_test, y_scores, window_sec=window_sec):
            row["patient"] = str(patient)
            curve_rows.append(row)

        if do_events:
            test_times = start_times[mask]
            intervals = seizure_intervals_by_patient.get(str(patient), [])
            event_metrics = compute_event_metrics(
                test_times, y_scores, intervals, event_threshold,
                window_sec=window_sec, hop_sec=hop_sec, **event_kwargs,
            )
            event_metrics["event_auprc"] = compute_event_auprc(
                test_times, y_scores, intervals, window_sec=window_sec, **event_kwargs,
            )
            event_metrics["patient"] = str(patient)
            event_metrics["skipped"] = ""
            event_metrics_results.append(event_metrics)

            for row in compute_event_curve(
                test_times, y_scores, intervals,
                window_sec=window_sec, hop_sec=hop_sec, **event_kwargs,
            ):
                row["patient"] = str(patient)
                event_curve_rows.append(row)

    return metrics_results, curve_rows, event_metrics_results, event_curve_rows
