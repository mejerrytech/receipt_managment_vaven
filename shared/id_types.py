"""Shared UUID helpers for database record identifiers."""

from __future__ import annotations

import uuid
from typing import Optional, Union

RecordId = uuid.UUID

IdLike = Union[str, uuid.UUID, None]


def as_uuid(value: IdLike) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def as_str(value: IdLike) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def sql_uuid(value: uuid.UUID) -> str:
    """PostgreSQL literal for embedding a UUID in raw SQL."""
    return f"'{value}'::uuid"
