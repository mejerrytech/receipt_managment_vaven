from uuid import UUID
from typing import Any, Optional

from api.review.utils.images import find_review_image, image_url
from api.review.utils.ocr import (
    confidence_tier,
    currency,
    field_confidence,
    gst_amount,
    gstin,
    invoice_date,
    invoice_number,
    merge_updates_into_ocr,
    nested_dict,
    notes,
    overall_confidence,
    parse_json,
    total_amount,
    vendor_name,
)
from shared.database import DatabaseService, PendingDocument, UserTextEntry
from shared.id_types import as_str

def field_detail(value: Any, confidence: float) -> dict[str, Any]:
    return {
        "value": value,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100),
        "tier": confidence_tier(confidence),
    }


def _tax_fields(pending: PendingDocument, data: dict) -> dict[str, Any]:
    tax = DatabaseService.extract_tax_fields_from_ocr(data)
    return {
        "gstin": pending.gstin or tax["gstin"],
        "gst_amount": pending.gst_amount if pending.gst_amount is not None else tax["gst_amount"],
        "igst_amount": pending.igst_amount if pending.igst_amount is not None else tax["igst_amount"],
        "cgst_amount": pending.cgst_amount if pending.cgst_amount is not None else tax["cgst_amount"],
        "sgst_amount": pending.sgst_amount if pending.sgst_amount is not None else tax["sgst_amount"],
    }


def receipt_summary(pending: PendingDocument, user_id: UUID) -> dict[str, Any]:
    data = parse_json(pending.extracted_data)
    overall = overall_confidence(data, pending)
    tax = _tax_fields(pending, data)
    cat = DatabaseService.normalize_expense_category_label(
        pending.expense_category or data.get("expense_category")
    )
    return {
        "id": as_str(pending.id),
        "token": pending.token,
        "vendor_name": vendor_name(data) or pending.file_name or "Unknown",
        "invoice_date": invoice_date(data),
        "total_amount": total_amount(data),
        "gstin": tax["gstin"],
        "gst_amount": tax["gst_amount"],
        "igst_amount": tax["igst_amount"],
        "cgst_amount": tax["cgst_amount"],
        "sgst_amount": tax["sgst_amount"],
        "currency": currency(data),
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "source": pending.source,
        "status": pending.status,
        "confidence_overall": overall,
        "confidence_tier": confidence_tier(overall),
        "user_id": as_str(user_id),
        "image_url": image_url(pending.id, user_id),
        "file_name": pending.file_name,
        "mime_type": pending.mime_type,
        "ocr_job_id": pending.ocr_job_id,
        "ocr_started_at": pending.ocr_started_at.isoformat() if pending.ocr_started_at else None,
        "ocr_completed_at": pending.ocr_completed_at.isoformat() if pending.ocr_completed_at else None,
        "created_at": pending.created_at.isoformat() if pending.created_at else None,
    }


def text_entry_summary(entry: UserTextEntry, user_id: UUID) -> dict[str, Any]:
    text = (entry.text or "").strip()
    cat = DatabaseService.normalize_expense_category_label(entry.expense_category)
    return {
        "id": as_str(entry.id),
        "token": None,
        "tab": "text",
        "record_type": "text",
        "vendor_name": text.split("\n", 1)[0][:80] if text else "Text Entry",
        "invoice_date": None,
        "total_amount": entry.amount,
        "gstin": None,
        "gst_amount": None,
        "igst_amount": None,
        "cgst_amount": None,
        "sgst_amount": None,
        "currency": entry.currency or "INR",
        "category": cat,
        "category_slug": DatabaseService.get_category_slug(cat),
        "source": entry.source or "telegram",
        "status": "approved",
        "confidence_overall": None,
        "confidence_tier": None,
        "user_id": as_str(user_id),
        "image_url": None,
        "file_name": None,
        "mime_type": None,
        "text": text,
        "user_input_text": text,
        "notes": text,
        "ocr_job_id": None,
        "ocr_started_at": None,
        "ocr_completed_at": None,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "intent_tag": entry.intent_tag,
    }


def receipt_detail(pending: PendingDocument, user_id: UUID) -> dict[str, Any]:
    data = parse_json(pending.extracted_data)
    overall = overall_confidence(data, pending)
    summary = receipt_summary(pending, user_id)
    summary["extracted_data"] = data
    summary["notes"] = notes(data, pending)
    summary["fields"] = {
        "vendor_name": field_detail(
            vendor_name(data),
            field_confidence(data, "vendor_name", overall),
        ),
        "gstin": field_detail(summary["gstin"], field_confidence(data, "gstin", overall)),
        "invoice_date": field_detail(
            invoice_date(data),
            field_confidence(data, "invoice_date", overall),
        ),
        "invoice_number": field_detail(
            invoice_number(data),
            field_confidence(data, "invoice_number", overall),
        ),
        "total_amount": field_detail(
            total_amount(data),
            field_confidence(data, "total_amount", overall),
        ),
        "gst_amount": field_detail(
            summary["gst_amount"],
            field_confidence(data, "gst_amount", overall),
        ),
        "igst_amount": field_detail(
            summary["igst_amount"],
            field_confidence(data, "igst_amount", overall),
        ),
        "cgst_amount": field_detail(
            summary["cgst_amount"],
            field_confidence(data, "cgst_amount", overall),
        ),
        "sgst_amount": field_detail(
            summary["sgst_amount"],
            field_confidence(data, "sgst_amount", overall),
        ),
        "currency": field_detail(currency(data), field_confidence(data, "currency", overall)),
        "category": field_detail(
            summary["category"],
            field_confidence(data, "expense_category", overall),
        ),
        "notes": field_detail(summary["notes"], overall or 0.0),
    }
    summary["has_image"] = find_review_image(pending.id) is not None or bool(pending.telegram_file_id)
    return summary


def matches_search(summary: dict[str, Any], query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    haystack = " ".join(
        str(v)
        for v in (
            summary.get("vendor_name"),
            summary.get("invoice_date"),
            summary.get("total_amount"),
            summary.get("gst_amount"),
            summary.get("category"),
            summary.get("file_name"),
            summary.get("notes"),
            summary.get("text"),
            summary.get("user_input_text"),
        )
        if v is not None
    ).lower()
    return q in haystack


def matches_confidence_tier(summary: dict[str, Any], tier: Optional[str]) -> bool:
    if not tier:
        return True
    return summary.get("confidence_tier") == tier.strip().lower()
