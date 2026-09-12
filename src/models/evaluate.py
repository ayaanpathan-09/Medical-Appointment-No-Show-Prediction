"""
Model evaluation utilities.

Computes the metrics required by the TRD (Section 4.3) and implements the
promotion rule from the FRD (Section 7): a new model may only be promoted
if its Recall and F1-score are >= the current production model's.
"""
from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model: Any, X_test, y_test) -> dict:
    """Return Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC (TRD 4.3)."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def can_promote(new_metrics: dict, production_metrics: dict | None) -> bool:
    """
    FRD Section 7 / FR-PE-06: a new model can only be promoted to production
    if its Recall and F1-score are equal to or better than the current
    production model's. If there is no production model yet, always allow.
    """
    if production_metrics is None:
        return True
    return (
        new_metrics["recall"] >= production_metrics["recall"]
        and new_metrics["f1_score"] >= production_metrics["f1_score"]
    )
