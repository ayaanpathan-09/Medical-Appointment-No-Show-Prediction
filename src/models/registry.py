"""
Model registry.

Stores versioned trained models and their evaluation metadata
(TRD Section 2.1 / Section 6: model_registry table), and tracks which
version is currently in production.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REGISTRY_DIR = Path("models_registry")
PRODUCTION_POINTER = "production.json"


def _registry_dir(base_dir: str | Path | None) -> Path:
    return Path(base_dir) if base_dir else DEFAULT_REGISTRY_DIR


def save_model(
    model: Any,
    version: str,
    algorithm: str,
    metrics: dict,
    feature_columns: list[str],
    base_dir: str | Path | None = None,
) -> Path:
    """Save a trained model + metadata under models_registry/<version>/."""
    reg_dir = _registry_dir(base_dir)
    version_dir = reg_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = version_dir / "model.pkl"
    joblib.dump(model, artifact_path)

    metadata = {
        "model_version": version,
        "algorithm": algorithm,
        "training_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "artifact_path": str(artifact_path),
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model version %s (%s) to %s", version, algorithm, version_dir)
    return version_dir


def load_model(version: str, base_dir: str | Path | None = None):
    reg_dir = _registry_dir(base_dir)
    version_dir = reg_dir / version
    model = joblib.load(version_dir / "model.pkl")
    with open(version_dir / "metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


def set_production_version(version: str, base_dir: str | Path | None = None) -> None:
    reg_dir = _registry_dir(base_dir)
    reg_dir.mkdir(parents=True, exist_ok=True)
    with open(reg_dir / PRODUCTION_POINTER, "w") as f:
        json.dump({"production_version": version}, f)
    logger.info("Promoted model version %s to production", version)


def get_production_version(base_dir: str | Path | None = None) -> str | None:
    reg_dir = _registry_dir(base_dir)
    pointer = reg_dir / PRODUCTION_POINTER
    if not pointer.exists():
        return None
    with open(pointer) as f:
        return json.load(f).get("production_version")


def load_production_model(base_dir: str | Path | None = None):
    version = get_production_version(base_dir)
    if version is None:
        raise FileNotFoundError(
            "No production model set. Run `python -m src.models.train <features.csv>` "
            "first -- it trains, evaluates, and promotes the best candidate automatically."
        )
    return load_model(version, base_dir)


def get_production_metrics(base_dir: str | Path | None = None) -> dict | None:
    version = get_production_version(base_dir)
    if version is None:
        return None
    _, metadata = load_model(version, base_dir)
    return metadata["metrics"]
