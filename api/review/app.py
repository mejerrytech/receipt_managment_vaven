from fastapi import FastAPI

from api.common.setup import register_global_response
from api.reports.routes import router as reports_router
from api.review.routes import router as review_router
from api.review.utils import ensure_image_dir
from shared.database import init_db
from shared.env import load_project_dotenv

load_project_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="Receipt Management API")
    register_global_response(app)
    app.include_router(review_router)
    app.include_router(reports_router)

    @app.on_event("startup")
    def _startup():
        init_db()
        ensure_image_dir()
        from api.reports.constants import REPORTS_DIR

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    return app


app = create_app()
