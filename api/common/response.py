from __future__ import annotations

from typing import Any, Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    code: int
    status: str
    message: str
    details: str = ""
    data: list[Any] = Field(default_factory=list)


def wrap_data(payload: Any) -> list[Any]:
    """Normalize any payload into the `data` list field."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    return [payload]


def success_response(
    data: Any = None,
    *,
    message: str = "Success",
    code: int = 200,
    details: str = "",
) -> JSONResponse:
    body = ApiResponse(
        code=code,
        status="success",
        message=message,
        details=details,
        data=wrap_data(data),
    )
    return JSONResponse(status_code=code, content=body.model_dump())


def error_response(
    message: str,
    *,
    code: int = 400,
    details: str = "",
    data: Any = None,
) -> JSONResponse:
    body = ApiResponse(
        code=code,
        status="error",
        message=message,
        details=details,
        data=wrap_data(data),
    )
    return JSONResponse(status_code=code, content=body.model_dump())


def is_envelope(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"code", "status", "message", "data"}
    return required.issubset(payload.keys())
