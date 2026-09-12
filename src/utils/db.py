"""
Database connection helpers (TRD Section 7: SQL database for structured data).

Uses SQLAlchemy so the same code works against SQLite (local dev, no setup
required) or MySQL/Postgres in staging/production via DATABASE_URL.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_engine = None
_SessionLocal = None


def get_engine():
    """Lazily create (and cache) the SQLAlchemy engine from DATABASE_URL."""
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL", "sqlite:///./noshow_dev.db")
        _engine = create_engine(url, future=True)
    return _engine


def get_session():
    """Return a new SQLAlchemy session bound to the shared engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal()
