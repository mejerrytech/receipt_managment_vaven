"""FastAPI application for dashboard and mobile backend APIs."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.database import init_db, engine, DATABASE_URL
from shared.env import load_project_dotenv
from api.routes.dashboard import router as dashboard_router

load_project_dotenv()

logger = logging.getLogger("api")

app = FastAPI(
    title="Expense Dashboard API",
    description="Backend APIs for spend dashboards, GST, calendar, and activity feeds.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)


@app.on_event("startup")
def _ensure_database_ready():
    init_db()
    logger.info("Database ready: %s (%s)", engine.dialect.name, DATABASE_URL.split("@")[-1])


@app.get("/health")
def health():
    return {"status": "ok", "service": "expense-dashboard-api"}
