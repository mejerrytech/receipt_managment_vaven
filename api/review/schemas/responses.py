from typing import Any, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    receipt: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class ActionResponse(BaseModel):
    success: bool
    message: str
    receipt: Optional[dict[str, Any]] = None
    document_id: Optional[int] = None


class RecentReceiptsResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    showing: int
    limit: int
    offset: int
    counts: dict[str, int]


class HistoryTabsResponse(BaseModel):
    document: int
    text: int


class DocumentHistoryResponse(BaseModel):
    tab: str
    tabs: HistoryTabsResponse
    items: list[dict[str, Any]]
    total: int
    page: int
    limit: int
    total_pages: int
    showing_from: int
    showing_to: int
    summary: str
