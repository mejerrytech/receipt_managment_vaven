"""Backward-compatible entry point — prefer api.review.app:app"""

from api.review.app import app, create_app
from api.review.routes import router

__all__ = ["app", "create_app", "router"]
