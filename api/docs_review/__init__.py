"""Backward-compatible shim — use api.review instead."""

from api.review.app import app, create_app
from api.review.routes import router

__all__ = ["app", "create_app", "router"]
