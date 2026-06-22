from __future__ import annotations

from uuid import UUID

from datetime import date, datetime
from typing import Any, Optional

from api.review.utils.ocr import gst_amount, parse_json
from shared.id_types import as_str
from shared.database import Document, get_db


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _doc_in_range(doc: Document, start: date, end: date) -> bool:
    if not doc.created_at:
        return False
    doc_day = doc.created_at.date() if hasattr(doc.created_at, "date") else None
    if doc_day is None:
        return False
    return start <= doc_day <= end


def _itc_eligible_gst(doc: Document, data: dict) -> float:
    gst = gst_amount(data) or 0.0
    gstin = (doc.gstin or "").strip()
    if not gstin or gst <= 0:
        return 0.0
    return float(gst)


def fetch_confirmed_receipts(
    user_id: UUID,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    session = get_db()
    try:
        docs = (
            session.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )
    finally:
        session.close()

    rows: list[dict[str, Any]] = []
    for doc in docs:
        if not _doc_in_range(doc, start, end):
            continue
        data = parse_json(doc.extracted_data)
        gst = gst_amount(data) or 0.0
        rows.append(
            {
                "id": as_str(doc.id),
                "date": doc.document_date or (
                    doc.created_at.strftime("%d %b %Y") if doc.created_at else ""
                ),
                "vendor": doc.vendor_name or doc.title or doc.file_name or "Unknown",
                "invoice_number": doc.invoice_number,
                "gstin": doc.gstin,
                "amount": doc.total_amount or 0.0,
                "gst_amount": gst,
                "itc_eligible": _itc_eligible_gst(doc, data),
                "category": doc.expense_category or "Other",
                "source": doc.source or "telegram",
                "currency": doc.currency or "INR",
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
        )
    return rows


def build_preview(
    user_id: UUID,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    rows = fetch_confirmed_receipts(user_id, start_date, end_date)
    total_amount = sum(r["amount"] for r in rows)
    total_gst = sum(r["gst_amount"] for r in rows)
    itc_eligible = sum(r["itc_eligible"] for r in rows)
    currency = rows[0]["currency"] if rows else "INR"

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    period_label = f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"

    return {
        "period_label": period_label,
        "start_date": start_date,
        "end_date": end_date,
        "total_invoices": len(rows),
        "total_amount": round(total_amount, 2),
        "total_gst": round(total_gst, 2),
        "itc_eligible": round(itc_eligible, 2),
        "currency": currency,
        "rows": rows,
    }


def category_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get("category") or "Other"
        bucket = buckets.setdefault(cat, {"count": 0, "amount": 0.0, "gst": 0.0})
        bucket["count"] += 1
        bucket["amount"] += float(row.get("amount") or 0)
        bucket["gst"] += float(row.get("gst_amount") or 0)
    return [
        {
            "category": cat,
            "count": int(v["count"]),
            "amount": round(v["amount"], 2),
            "gst": round(v["gst"], 2),
        }
        for cat, v in sorted(buckets.items(), key=lambda x: x[1]["amount"], reverse=True)
    ]


def itc_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in rows if (r.get("itc_eligible") or 0) > 0]
    ineligible = [r for r in rows if (r.get("itc_eligible") or 0) <= 0 and (r.get("gst_amount") or 0) > 0]
    return {
        "itc_eligible_total": round(sum(r["itc_eligible"] for r in eligible), 2),
        "eligible_invoices": len(eligible),
        "ineligible_invoices": len(ineligible),
        "total_gst": round(sum(r["gst_amount"] for r in rows), 2),
    }
