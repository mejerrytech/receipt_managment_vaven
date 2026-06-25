"""Shared FastAPI dependencies."""

from fastapi import HTTPException, Query

from shared.database import DatabaseService, User


def get_user_id(user_id: int = Query(..., description="Internal users.id")) -> int:
    """Resolve and validate user_id query parameter."""
    user = DatabaseService.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.id


def get_optional_user(user_id: int = Query(..., description="Internal users.id")) -> User:
    user = DatabaseService.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
