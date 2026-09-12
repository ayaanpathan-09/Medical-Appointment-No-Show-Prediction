"""
Centralized logging setup, configurable via config/logging.yaml.
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

import yaml

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    config_path = Path("config/logging.yaml")
    if config_path.exists():
        with open(config_path) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use."""
    _configure()
    return logging.getLogger(name)
