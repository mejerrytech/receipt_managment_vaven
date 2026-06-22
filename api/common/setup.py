import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.common.middleware import GlobalResponseMiddleware
from api.common.response import ApiResponse, wrap_data

logger = logging.getLogger("api.setup")


def register_global_response(app: FastAPI) -> None:
    """Attach global response middleware and exception handlers."""
    app.add_middleware(GlobalResponseMiddleware)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else "Request failed"
        details = ""
        if isinstance(detail, (list, dict)):
            details = str(detail)
            message = "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiResponse(
                code=exc.status_code,
                status="error",
                message=message,
                details=details,
                data=[],
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        return JSONResponse(
            status_code=422,
            content=ApiResponse(
                code=422,
                status="error",
                message="Validation error",
                details=str(errors),
                data=wrap_data(errors),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(
                code=500,
                status="error",
                message="Internal server error",
                details=str(exc),
                data=[],
            ).model_dump(),
        )
