"""API-key gate (placeholder — NOT production auth).

When ``SERVICE_API_KEY`` is unset, the gate is a no-op. When set, the
``X-API-Key`` header on each request must match. This is a demo-time
guard against accidental exposure of the Anthropic API budget; real
auth would use OAuth / JWT / a secret-manager-backed rotation scheme.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate X-API-Key if SERVICE_API_KEY is set.

    Raises:
        HTTPException 401 if key is required but missing/wrong.
    """
    expected = os.environ.get("SERVICE_API_KEY")
    if not expected:
        return  # auth disabled
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
