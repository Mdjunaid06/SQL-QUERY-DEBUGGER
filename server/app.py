# server/app.py — required by OpenEnv validator
# This imports and re-exports the main FastAPI app

from api.server import app

__all__ = ["app"]