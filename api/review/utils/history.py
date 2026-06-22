from __future__ import annotations

from uuid import UUID

from datetime import date, datetime
from math import ceil
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
from api.review.utils.recent import format_display_date, review_status_from_confidence
from shared.id_types import as_str
from shared.database import (
    DatabaseService,
    Document,
    PendingDocument,
    UserTextEntry,
    get_db,
)

HistoryTab = Literal["document", "text"]

TIER_LABELS = {
    "green": "Green",
    "amber": "Amber",
    "red": "Red",
    "grey": "Grey",
}

SOURCE_LABELS = {
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
    "web": "Web",
}


def format_table_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        return value.strftime("%d %b %Y")
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text[:20]
    return dt.strftime("%d %b %Y")


def source_label(source: Optional[str]) -> str:
    key = (source or "telegram").strip().lower()
    return SOURCE_LABELS.get(key, key.capitalize())


def tier_label(tier: Optional[str]) -> Optional[str]:
    if not tier:
        return None
    return TIER_LABELS.get(tier.lower(), tier.capitalize())


def _parse_filter_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _item_created_date(item: dict[str, Any]) -> Optional[date]:
    raw = item.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _matches_search(item: dict[str, Any], query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    parts = [
        item.get("vendor"),
        item.get("text"),
        item.get("notes"),
        item.get("category"),
        item.get("amount"),
        item.get("gst_amount"),
        item.get("source"),
        item.get("date"),
    ]
    haystack = " ".join(str(p) for p in parts if p is not None).lower()
    return q in haystack


def _matches_date_range(
    item: dict[str, Any],
    start_date: Optional[date],
    end_date: Optional[date],
) -> bool:
    if not start_date and not end_date:
        return True
    item_date = _item_created_date(item)
    if item_date is None:
        return False
    if start_date and item_date < start_date:
        return False
    if end_date and item_date > end_date:
        return False
    return True


def _apply_filters(
    items: list[dict[str, Any]],
    *,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    tier: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    start = _parse_filter_date(start_date)
    end = _parse_filter_date(end_date)

    if category:
        norm = DatabaseService.resolve_category_filter(category)
        items = [i for i in items if i.get("category") == norm]
    if source:
        src = source.strip().lower()
        items = [i for i in items if (i.get("source") or "").lower() == src]
    if tier:
        tier_norm = tier.strip().lower()
        items = [i for i in items if (i.get("tier") or "").lower() == tier_norm]
    if status:
        status_norm = status.strip().lower()
        items = [i for i in items if (i.get("status") or "").lower() == status_norm]
    if search:
        items = [i for i in items if _matches_search(i, search)]
    if start or end:
        items = [i for i in items if _matches_date_range(i, start, end)]
    return items


def history_from_document(doc: Document, user_id: UUID) -> dict[str, Any]:
    data = parse_json(doc.extracted_data)
    tier = confidence_tier(doc.confidence_overall)
    row_date = doc.document_date or format_table_date(doc.created_at)
    cat = DatabaseService.normalize_expense_category_label(doc.expense_category)
    return {
        "id": as_str(doc.id),
        "record_type": "document",
        "tab": "document",
        "date": row_date,
        "display_date": format_display_date(doc.created_at),
        "vendor": doc.vendor_name or vendor_name(data) or doc.title or doc.file_name or "Receipt",
        "amount": doc.total_amount,
        "gst_amount": gst_amount(data),
        "currency": doc.currency or currency(data),
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "source": doc.source or "telegram",
        "source_label": source_label(doc.source),
        "tier": tier,
        "tier_label": tier_label(tier),
        "status": "approved",
        "status_label": "Approved",
        "confidence_overall": doc.confidence_overall,
        "notes": doc.user_input_text,
        "text": None,
        "image_url": None,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


def history_from_pending(pending: PendingDocument, user_id: UUID) -> dict[str, Any]:
    data = parse_json(pending.extracted_data)
    overall = overall_confidence(data, pending)
    status, status_label = review_status_from_confidence(overall)
    tier = confidence_tier(overall)
    row_date = invoice_date(data) or format_table_date(pending.created_at)
    cat = DatabaseService.normalize_expense_category_label(
        pending.expense_category or data.get("expense_category")
    )
    return {
        "id": as_str(pending.id),
        "record_type": "pending",
        "tab": "document",
        "date": row_date,
        "display_date": format_display_date(pending.created_at),
        "vendor": vendor_name(data) or pending.file_name or "Receipt",
        "amount": total_amount(data),
        "gst_amount": gst_amount(data),
        "currency": currency(data),
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "source": pending.source,
        "source_label": source_label(pending.source),
        "tier": tier,
        "tier_label": tier_label(tier),
        "status": status,
        "status_label": status_label,
        "confidence_overall": overall,
        "notes": pending.user_input_text,
        "text": None,
        "image_url": image_url(pending.id, user_id),
        "created_at": pending.created_at.isoformat() if pending.created_at else None,
    }


def history_from_text(entry: UserTextEntry) -> dict[str, Any]:
    text = (entry.text or "").strip()
    cat = DatabaseService.normalize_expense_category_label(entry.expense_category)
    return {
        "id": as_str(entry.id),
        "record_type": "text",
        "tab": "text",
        "date": format_table_date(entry.created_at),
        "display_date": format_display_date(entry.created_at),
        "vendor": text.split("\n", 1)[0][:80] if text else "Text Entry",
        "amount": entry.amount,
        "gst_amount": None,
        "currency": entry.currency or "INR",
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "source": entry.source or "telegram",
        "source_label": source_label(entry.source),
        "tier": "grey",
        "tier_label": "Grey",
        "status": "manual_entry",
        "status_label": "Manual Entry",
        "confidence_overall": None,
        "notes": None,
        "text": text,
        "image_url": None,
        "intent_tag": entry.intent_tag,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


def _load_document_history_rows(user_id: UUID) -> list[dict[str, Any]]:
    session = get_db()
    try:
        docs = (
            session.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )
        pendings = (
            session.query(PendingDocument)
            .filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(list(REVIEW_STATUSES)),
            )
            .order_by(PendingDocument.created_at.desc())
            .all()
        )
    finally:
        session.close()

    items: list[dict[str, Any]] = []
    items.extend(history_from_document(d, user_id) for d in docs)
    items.extend(history_from_pending(p, user_id) for p in pendings)
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items


def _load_text_history_rows(user_id: UUID) -> list[dict[str, Any]]:
    session = get_db()
    try:
        texts = (
            session.query(UserTextEntry)
            .filter(UserTextEntry.user_id == user_id)
            .order_by(UserTextEntry.created_at.desc())
            .all()
        )
    finally:
        session.close()
    return [history_from_text(t) for t in texts]


def _tab_counts(user_id: UUID) -> dict[str, int]:
    session = get_db()
    try:
        doc_count = session.query(Document).filter(Document.user_id == user_id).count()
        pending_count = (
            session.query(PendingDocument)
            .filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(list(REVIEW_STATUSES)),
            )
            .count()
        )
        text_count = session.query(UserTextEntry).filter(UserTextEntry.user_id == user_id).count()
    finally:
        session.close()
    return {
        "document": doc_count + pending_count,
        "text": text_count,
    }


def fetch_document_history(
    user_id: UUID,
    *,
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    tier: Optional[str] = None,
    status: Optional[str] = None,
) -> dict[str, Any]:
    items = _load_document_history_rows(user_id)
    items = _apply_filters(
        items,
        search=search,
        start_date=start_date,
        end_date=end_date,
        category=category,
        source=source,
        tier=tier,
        status=status,
    )
    return _paginate(items, tab="document", user_id=user_id, page=page, limit=limit)


def fetch_text_history(
    user_id: UUID,
    *,
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    tier: Optional[str] = None,
    status: Optional[str] = None,
) -> dict[str, Any]:
    items = _load_text_history_rows(user_id)
    items = _apply_filters(
        items,
        search=search,
        start_date=start_date,
        end_date=end_date,
        category=category,
        source=source,
        tier=tier,
        status=status,
    )
    return _paginate(items, tab="text", user_id=user_id, page=page, limit=limit)


def fetch_history(
    user_id: UUID,
    tab: HistoryTab,
    *,
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    tier: Optional[str] = None,
    status: Optional[str] = None,
) -> dict[str, Any]:
    if tab == "text":
        return fetch_text_history(
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
    return fetch_document_history(
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


def _paginate(
    items: list[dict[str, Any]],
    *,
    tab: HistoryTab,
    user_id: UUID,
    page: int,
    limit: int,
) -> dict[str, Any]:
    total = len(items)
    total_pages = ceil(total / limit) if total else 0
    offset = (page - 1) * limit
    page_items = items[offset : offset + limit]
    showing_from = offset + 1 if page_items else 0
    showing_to = offset + len(page_items)

    return {
        "tab": tab,
        "tabs": _tab_counts(user_id),
        "items": page_items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "showing_from": showing_from,
        "showing_to": showing_to,
        "summary": f"Showing {showing_from} to {showing_to} of {total} receipts",
    }
