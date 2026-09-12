"""
Authentication for the Scoring API (FR-UA-01 / FR-UA-02).

A minimal API-key scheme, sufficient for a college project. Swap this out
for OAuth2 / your institution's identity provider before any real
production use (TRD Section 5.3: "Authentication required (API key or
OAuth token) for all endpoints.").
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def _valid_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "dev-local-key")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key not in _valid_keys():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return x_api_key
