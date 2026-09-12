"""
Model training entry point.

Trains and compares the candidate algorithms defined in the TRD
(Section 4.1) against src/models/config/model_params.yaml, selects the best
model by the Recall/F1 trade-off (TRD Section 4.3), and registers it via
src/models/registry.py. Implements the promotion gate from FRD Section 7 /
FR-PE-06.

Run with:
    python -m src.models.train data/processed/features.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib

import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.models import registry
from src.models.evaluate import can_promote, evaluate_model
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def split_data(X, y, cfg: dict):
    """Stratified 70/15/15 train/val/test split (TRD Section 4.2)."""
    val_size = cfg["val_size"]
    test_size = cfg["test_size"]
    random_state = cfg["random_state"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val, stratify=y_train_val, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def maybe_resample(X_train, y_train, cfg: dict):
    """Address class imbalance via SMOTE (TRD Section 4.2), if enabled and available."""
    if not cfg.get("use_smote"):
        return X_train, y_train
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed; skipping SMOTE, relying on class_weight only.")
        return X_train, y_train

    if len(X_train) > 50000:
        logger.info(
            "Dataset has %d training rows; skipping SMOTE to avoid excessive memory/time use. "
            "Class-weighted models will handle imbalance.", len(X_train)
        )
        return X_train, y_train

    minority_count = y_train.value_counts().min()
    if minority_count < 2:
        logger.warning(
            "Minority class has only %d sample(s); skipping SMOTE, relying on class_weight only.",
            minority_count,
        )
        return X_train, y_train

    # SMOTE requires k_neighbors < minority_count. Scale down automatically for
    # small datasets (e.g. the bundled sample data) instead of failing.
    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=cfg["random_state"], k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info("Applied SMOTE (k_neighbors=%d): %d -> %d training rows", k_neighbors, len(X_train), len(X_res))
    return X_res, y_res


def train_and_compare(features_path: str, params_path: str, registry_dir: str | None = None) -> dict:
    df = pd.read_csv(features_path)
    if df.empty:
        raise ValueError(
            "No training rows are available in features.csv. "
            "Check data/processed/ingestion_audit.json for rejected rows and run the pipeline again."
        )
    if "no_show" not in df.columns:
        raise ValueError("Training data must include the 'no_show' target column.")

    y = df["no_show"].astype(int)
    X = df.drop(columns=["appointment_id", "no_show"], errors="ignore")
    feature_columns = list(X.columns)

    all_cfg = _load_config(params_path)
    train_cfg = all_cfg.pop("training")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_cfg)
    X_train_res, y_train_res = maybe_resample(X_train, y_train, train_cfg)

    cv = StratifiedKFold(
        n_splits=train_cfg["cv_folds"],
        shuffle=True,
        random_state=train_cfg["random_state"]
    )

    # Skip SVM for large datasets because SVC + GridSearchCV
    # is extremely slow for large datasets.
    if len(X_train_res) > 50000:
        all_cfg.pop("svm", None)
        logger.info("Large dataset detected; skipping SVM training.")

    results = {}

    for algo_name, algo_cfg in all_cfg.items():
        logger.info("Training candidate: %s", algo_name)
        model_class = _load_class(algo_cfg["class"])
        base_model = model_class()
        search = GridSearchCV(
            base_model, algo_cfg["params"], scoring="f1", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train_res, y_train_res)
        val_metrics = evaluate_model(search.best_estimator_, X_val, y_val)
        results[algo_name] = {
            "model": search.best_estimator_,
            "best_params": search.best_params_,
            "val_metrics": val_metrics,
        }
        logger.info("%s val metrics: %s", algo_name, val_metrics)

    # Select best by Recall/F1 trade-off (TRD 4.3): primary key F1, tie-break Recall.
    best_name = max(
        results,
        key=lambda name: (results[name]["val_metrics"]["f1_score"], results[name]["val_metrics"]["recall"]),
    )
    best_model = results[best_name]["model"]
    test_metrics = evaluate_model(best_model, X_test, y_test)
    logger.info("Best candidate: %s | test metrics: %s", best_name, test_metrics)

    production_metrics = registry.get_production_metrics(registry_dir)
    promote = can_promote(test_metrics, production_metrics)

    version = dt.datetime.now(dt.timezone.utc).strftime("v%Y%m%d_%H%M%S")
    registry.save_model(best_model, version, best_name, test_metrics, feature_columns, registry_dir)

    if promote:
        registry.set_production_version(version, registry_dir)
        logger.info("Model %s promoted to production.", version)
    else:
        logger.info(
            "Model %s NOT promoted (Recall/F1 below current production: %s).",
            version, production_metrics,
        )

    return {
        "candidates": {name: r["val_metrics"] for name, r in results.items()},
        "selected": best_name,
        "version": version,
        "test_metrics": test_metrics,
        "promoted": promote,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare no-show prediction models.")
    parser.add_argument("features_csv", help="Path to engineered features CSV (must include no_show)")
    parser.add_argument("--params", default="src/models/config/model_params.yaml")
    parser.add_argument("--registry-dir", default="models_registry")
    args = parser.parse_args()

    summary = train_and_compare(args.features_csv, args.params, args.registry_dir)
    print(summary)
