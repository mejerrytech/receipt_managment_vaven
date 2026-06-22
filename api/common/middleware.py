import json
import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.common.response import ApiResponse, is_envelope, wrap_data

logger = logging.getLogger("api.middleware")

SUCCESS_MESSAGES = {
    "GET": "Data fetched successfully",
    "POST": "Created successfully",
    "PUT": "Updated successfully",
    "PATCH": "Updated successfully",
    "DELETE": "Deleted successfully",
}


class GlobalResponseMiddleware(BaseHTTPMiddleware):
    """Wrap all /api JSON responses in the global envelope."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        response = await call_next(request)
        content_type = (response.headers.get("content-type") or "").lower()

        if "application/json" not in content_type:
            return response

        body_bytes = b""
        async for chunk in response.body_iterator:
            body_bytes += chunk

        if not body_bytes:
            envelope = ApiResponse(
                code=response.status_code,
                status="success" if response.status_code < 400 else "error",
                message=SUCCESS_MESSAGES.get(request.method, "Success"),
                details="",
                data=[],
            )
            return JSONResponse(
                status_code=response.status_code,
                content=envelope.model_dump(),
                headers=_passthrough_headers(response),
            )

        try:
            payload: Any = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        if is_envelope(payload):
            return JSONResponse(
                status_code=response.status_code,
                content=payload,
                headers=_passthrough_headers(response),
            )

        if response.status_code >= 400:
            message = _error_message(payload)
            envelope = ApiResponse(
                code=response.status_code,
                status="error",
                message=message,
                details=_error_details(payload, message),
                data=wrap_data(payload.get("data")) if isinstance(payload, dict) else [],
            )
        else:
            envelope = ApiResponse(
                code=response.status_code,
                status="success",
                message=_success_message(request, payload),
                details="",
                data=wrap_data(payload),
            )

        return JSONResponse(
            status_code=response.status_code,
            content=envelope.model_dump(),
            headers=_passthrough_headers(response),
        )


def _passthrough_headers(response: Response) -> dict[str, str]:
    skip = {"content-length", "content-type"}
    return {k: v for k, v in response.headers.items() if k.lower() not in skip}


def _success_message(request: Request, payload: Any) -> str:
    if isinstance(payload, dict):
        if payload.get("message"):
            return str(payload["message"])
        if payload.get("success") is True:
            return "Operation completed successfully"
    return SUCCESS_MESSAGES.get(request.method, "Success")


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("detail"), str):
            return payload["detail"]
        if isinstance(payload.get("message"), str):
            return payload["message"]
        if isinstance(payload.get("error"), str):
            return payload["error"]
    return "Request failed"


def _error_details(payload: Any, message: str) -> str:
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail")
    if isinstance(detail, list):
        return json.dumps(detail, ensure_ascii=False)
    if isinstance(detail, dict):
        return json.dumps(detail, ensure_ascii=False)
    extra = payload.get("details") or payload.get("error")
    if extra and str(extra) != message:
        return str(extra)
    return ""
