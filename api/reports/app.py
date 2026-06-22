from fastapi import FastAPI

from api.common.setup import register_global_response
from api.reports.constants import REPORTS_DIR
from api.reports.routes import router
from shared.database import init_db
from shared.env import load_project_dotenv

load_project_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="Reports API")
    register_global_response(app)
    app.include_router(router)

    @app.on_event("startup")
    def _startup():
        init_db()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    return app


app = create_app()
