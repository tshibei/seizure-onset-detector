"""Evaluation metrics for seizure onset detection."""
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve


def compute_metrics(y_true, y_pred, y_scores, window_sec=1.0):
    """Compute sensitivity, FPR/hour, and AUPRC.

    AUPRC is NaN if y_true is single-class (curve is undefined).
    Forcing labels=[0, 1] on confusion_matrix keeps the 2x2 shape even when
    a fold's predictions or labels collapse to one class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_test = len(y_true)
    n_test_pos = int((y_true == 1).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    total_hours = (n_test * window_sec) / 3600
    fpr_per_hour = fp / total_hours if total_hours > 0 else float("nan")

    auprc_by_chance = n_test_pos / n_test if n_test > 0 else float("nan")

    if 0 < n_test_pos < n_test:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall, precision)
    else:
        auprc = float("nan")

    return {
        "sensitivity": sensitivity,
        "fpr_per_hour": fpr_per_hour,
        "auprc": auprc,
        "auprc_by_chance": auprc_by_chance,
        "n_test": n_test,
        "n_test_pos": n_test_pos,
    }


def leave_one_patient_out_cv(X, y, patient_ids, model_class, window_sec=1.0):
    """Perform leave-one-patient-out cross-validation."""
    patient_ids = np.asarray(patient_ids)
    results = []

    for patient in np.unique(patient_ids):
        train_mask = patient_ids != patient
        test_mask = patient_ids == patient

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2:
            n_test = len(y_test)
            n_test_pos = int((np.asarray(y_test) == 1).sum())
            results.append({
                "patient": str(patient),
                "sensitivity": float("nan"),
                "fpr_per_hour": float("nan"),
                "auprc": float("nan"),
                "auprc_by_chance": n_test_pos / n_test if n_test > 0 else float("nan"),
                "n_test": n_test,
                "n_test_pos": n_test_pos,
                "skipped": "single_class_train",
            })
            continue

        model = model_class(X_train, y_train)
        y_pred = model.predict(X_test)

        proba = model.predict_proba(X_test)
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = model.named_steps[list(model.named_steps)[-1]].classes_
        pos_col = np.where(classes == 1)[0]
        y_scores = proba[:, pos_col[0]] if len(pos_col) else np.zeros(len(X_test))

        metrics = compute_metrics(y_test, y_pred, y_scores, window_sec=window_sec)
        metrics["patient"] = str(patient)
        results.append(metrics)

    return results
