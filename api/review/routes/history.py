from uuid import UUID
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query

from api.review.dependencies import require_user_id
from api.review.schemas import DocumentHistoryResponse, HistoryTabsResponse
from api.review.utils.history import fetch_document_history, fetch_history, fetch_text_history

router = APIRouter(prefix="/api/review/history", tags=["history"])

HistoryTab = Literal["document", "text"]


@router.get("/tabs", response_model=HistoryTabsResponse)
def history_tab_counts(user_id: UUID = Depends(require_user_id)) -> HistoryTabsResponse:
    """Counts for Document History and Text History tabs."""
    from api.review.utils.history import _tab_counts

    counts = _tab_counts(user_id)
    return HistoryTabsResponse(**counts)


@router.get("", response_model=DocumentHistoryResponse)
def list_history(
    user_id: UUID = Depends(require_user_id),
    tab: HistoryTab = Query("document", description="document = image receipts, text = manual entries"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by vendor, amount, notes..."),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    category: Optional[str] = Query(None, description="Category name or slug (e.g. groceries)"),
    source: Optional[str] = Query(None, description="telegram, whatsapp, or web"),
    tier: Optional[str] = Query(None, description="green, amber, red, grey"),
    status: Optional[str] = Query(
        None,
        description="approved, pending_review, review_required, manual_entry",
    ),
) -> DocumentHistoryResponse:
    """
    Document History dashboard — two tabs:
    - **document**: saved receipts + pending image uploads
    - **text**: user text/manual expense entries
    """
    data = fetch_history(
        user_id,
        tab,
        page=page,
        limit=limit,
        search=search,
        start_date=start_date,
        end_date=end_date,
        category=category,
        source=source,
        tier=tier,
        status=status,
    )
    return DocumentHistoryResponse(
        tab=data["tab"],
        tabs=HistoryTabsResponse(**data["tabs"]),
        items=data["items"],
        total=data["total"],
        page=data["page"],
        limit=data["limit"],
        total_pages=data["total_pages"],
        showing_from=data["showing_from"],
        showing_to=data["showing_to"],
        summary=data["summary"],
    )


@router.get("/documents", response_model=DocumentHistoryResponse)
def list_document_history(
    user_id: UUID = Depends(require_user_id),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    category: Optional[str] = Query(None, description="Category name or slug (e.g. groceries)"),
    source: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> DocumentHistoryResponse:
    """Document History tab — image/OCR receipts only (saved + pending)."""
    data = fetch_document_history(
        user_id,
        page=page,
        limit=limit,
        search=search,
        start_date=start_date,
        end_date=end_date,
        category=category,
        source=source,
        tier=tier,
        status=status,
    )
    return DocumentHistoryResponse(
        tab=data["tab"],
        tabs=HistoryTabsResponse(**data["tabs"]),
        items=data["items"],
        total=data["total"],
        page=data["page"],
        limit=data["limit"],
        total_pages=data["total_pages"],
        showing_from=data["showing_from"],
        showing_to=data["showing_to"],
        summary=data["summary"],
    )


@router.get("/texts", response_model=DocumentHistoryResponse)
def list_text_history(
    user_id: UUID = Depends(require_user_id),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    category: Optional[str] = Query(None, description="Category name or slug (e.g. groceries)"),
    source: Optional[str] = Query(None),
    tier: Optional[str] = Query(None, description="grey for manual entries"),
    status: Optional[str] = Query(None, description="manual_entry"),
) -> DocumentHistoryResponse:
    """Text History tab — manual user text entries only."""
    data = fetch_text_history(
        user_id,
        page=page,
        limit=limit,
        search=search,
        start_date=start_date,
        end_date=end_date,
        category=category,
        source=source,
        tier=tier,
        status=status,
    )
    return DocumentHistoryResponse(
        tab=data["tab"],
        tabs=HistoryTabsResponse(**data["tabs"]),
        items=data["items"],
        total=data["total"],
        page=data["page"],
        limit=data["limit"],
        total_pages=data["total_pages"],
        showing_from=data["showing_from"],
        showing_to=data["showing_to"],
        summary=data["summary"],
    )
