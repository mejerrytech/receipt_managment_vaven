from uuid import UUID

from datetime import datetime
from typing import Any, Literal, Optional

from api.review.constants import REVIEW_STATUSES
from api.review.utils.images import image_url
from api.review.utils.ocr import (
    confidence_tier,
    currency,
    gst_amount,
    invoice_date,
    overall_confidence,
    parse_json,
    total_amount,
    vendor_name,
)
from shared.id_types import as_str
from shared.database import (
    DatabaseService,
    Document,
    PendingDocument,
    UserTextEntry,
    get_db,
)


def format_display_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{dt.day} {dt.strftime('%b %Y')} • {hour}:{dt.strftime('%M %p')}"


def review_status_from_confidence(confidence: Optional[float]) -> tuple[str, str]:
    if confidence is None or confidence < 0.80:
        return "review_required", "Review Required"
    return "pending_review", "Pending Review"


def recent_from_document(doc: Document, user_id: UUID) -> dict[str, Any]:
    data = parse_json(doc.extracted_data)
    cat = DatabaseService.normalize_expense_category_label(doc.expense_category)
    return {
        "id": as_str(doc.id),
        "type": "receipt",
        "record_type": "document",
        "tab": "document",
        "vendor_name": doc.vendor_name or vendor_name(data) or doc.title or doc.file_name or "Receipt",
        "title": doc.title or doc.file_name,
        "text": None,
        "user_input_text": doc.user_input_text,
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "total_amount": doc.total_amount,
        "gst_amount": gst_amount(data),
        "currency": doc.currency or currency(data),
        "source": doc.source or "telegram",
        "status": "approved",
        "status_label": "Approved",
        "confidence_overall": doc.confidence_overall,
        "confidence_tier": confidence_tier(doc.confidence_overall),
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "display_date": format_display_date(doc.created_at),
        "invoice_date": doc.document_date or invoice_date(data),
        "image_url": None,
    }


def recent_from_pending(pending: PendingDocument, user_id: UUID) -> dict[str, Any]:
    data = parse_json(pending.extracted_data)
    overall = overall_confidence(data, pending)
    status, status_label = review_status_from_confidence(overall)
    cat = DatabaseService.normalize_expense_category_label(
        pending.expense_category or data.get("expense_category")
    )
    return {
        "id": as_str(pending.id),
        "type": "receipt",
        "record_type": "pending",
        "tab": "document",
        "vendor_name": vendor_name(data) or pending.file_name or "Receipt",
        "title": pending.file_name,
        "text": None,
        "user_input_text": pending.user_input_text,
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "total_amount": total_amount(data),
        "gst_amount": gst_amount(data),
        "currency": currency(data),
        "source": pending.source,
        "status": status,
        "status_label": status_label,
        "confidence_overall": overall,
        "confidence_tier": confidence_tier(overall),
        "created_at": pending.created_at.isoformat() if pending.created_at else None,
        "display_date": format_display_date(pending.created_at),
        "invoice_date": invoice_date(data),
        "image_url": image_url(pending.id, user_id),
    }


def recent_from_text(entry: UserTextEntry) -> dict[str, Any]:
    text = (entry.text or "").strip()
    cat = DatabaseService.normalize_expense_category_label(entry.expense_category)
    return {
        "id": as_str(entry.id),
        "type": "text",
        "record_type": "text",
        "tab": "text",
        "vendor_name": text.split("\n", 1)[0][:80] if text else "Text Entry",
        "title": text[:120] if text else "Text Entry",
        "text": text,
        "user_input_text": text,
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "total_amount": entry.amount,
        "gst_amount": None,
        "currency": entry.currency or "INR",
        "source": entry.source or "telegram",
        "status": "approved",
        "status_label": "Approved",
        "confidence_overall": None,
        "confidence_tier": None,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "display_date": format_display_date(entry.created_at),
        "invoice_date": None,
        "image_url": None,
        "intent_tag": entry.intent_tag,
    }


def fetch_recent_receipts(
    user_id: UUID,
    *,
    limit: int,
    offset: int,
    tab: Optional[Literal["document", "text"]] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
) -> dict[str, Any]:
    fetch_n = min(limit + offset, 500)
    session = get_db()
    try:
        docs = (
            session.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(fetch_n)
            .all()
        )
        pendings = (
            session.query(PendingDocument)
            .filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(list(REVIEW_STATUSES)),
            )
            .order_by(PendingDocument.created_at.desc())
            .limit(fetch_n)
            .all()
        )
        texts = (
            session.query(UserTextEntry)
            .filter(UserTextEntry.user_id == user_id)
            .order_by(UserTextEntry.created_at.desc())
            .limit(fetch_n)
            .all()
        )

        total_documents = session.query(Document).filter(Document.user_id == user_id).count()
        total_pending = (
            session.query(PendingDocument)
            .filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(list(REVIEW_STATUSES)),
            )
            .count()
        )
        total_texts = session.query(UserTextEntry).filter(UserTextEntry.user_id == user_id).count()
    finally:
        session.close()

    items: list[dict[str, Any]] = []
    if tab in (None, "document"):
        items.extend(recent_from_document(d, user_id) for d in docs)
        items.extend(recent_from_pending(p, user_id) for p in pendings)
    if tab in (None, "text"):
        items.extend(recent_from_text(t) for t in texts)
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)

    if category:
        norm = DatabaseService.resolve_category_filter(category)
        items = [i for i in items if i["category"] == norm]
    if source:
        src = source.strip().lower()
        items = [i for i in items if (i.get("source") or "").lower() == src]
    if status:
        status_norm = status.strip().lower()
        items = [i for i in items if i["status"] == status_norm]

    total = total_documents + total_pending + total_texts
    page = items[offset : offset + limit]
    return {
        "items": page,
        "total": total,
        "showing": len(page),
        "limit": limit,
        "offset": offset,
        "counts": {
            "documents": total_documents,
            "pending": total_pending,
            "texts": total_texts,
        },
    }
