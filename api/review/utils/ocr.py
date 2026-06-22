import json
from typing import Any, Optional

from shared.database import PendingDocument


def parse_json(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def nested_dict(data: dict, key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def confidence_tier(score: Optional[float]) -> str:
    if score is None:
        return "red"
    if score >= 0.95:
        return "green"
    if score >= 0.80:
        return "amber"
    return "red"


def field_confidence(data: dict, field_key: str, overall: Optional[float]) -> float:
    confidence = nested_dict(data, "confidence")
    raw = confidence.get(field_key)
    if raw is None:
        return overall if overall is not None else 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return overall if overall is not None else 0.0


def vendor_name(data: dict) -> Optional[str]:
    vendor = nested_dict(data, "vendor_or_sender")
    name = vendor.get("name") or data.get("vendor_name")
    return str(name).strip() if name else None


def invoice_date(data: dict) -> Optional[str]:
    dates = nested_dict(data, "dates")
    for key in ("invoice_date", "date", "document_date", "bill_date"):
        if dates.get(key):
            return str(dates[key])
    if data.get("document_date"):
        return str(data["document_date"])
    return None


def invoice_number(data: dict) -> Optional[str]:
    identifiers = nested_dict(data, "identifiers")
    value = identifiers.get("invoice_number") or data.get("bill_number") or data.get("invoice_number")
    return str(value).strip() if value else None


def gstin(data: dict) -> Optional[str]:
    from shared.database import DatabaseService

    return DatabaseService.extract_tax_fields_from_ocr(data).get("gstin")


def total_amount(data: dict) -> Optional[float]:
    amounts = nested_dict(data, "amounts")
    raw = amounts.get("total")
    if raw is None:
        raw = data.get("total_amount")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def currency(data: dict) -> str:
    amounts = nested_dict(data, "amounts")
    return str(amounts.get("currency") or data.get("currency") or "INR")


def gst_amount(data: dict) -> Optional[float]:
    from shared.database import DatabaseService

    return DatabaseService.extract_tax_fields_from_ocr(data).get("gst_amount")


def notes(data: dict, pending: PendingDocument) -> Optional[str]:
    if pending.user_input_text:
        return pending.user_input_text
    return data.get("notes") or data.get("user_notes")


def overall_confidence(data: dict, pending: PendingDocument) -> Optional[float]:
    if pending.confidence_overall is not None:
        return float(pending.confidence_overall)
    confidence = nested_dict(data, "confidence")
    raw = confidence.get("overall")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def merge_updates_into_ocr(data: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    if updates.get("vendor_name") is not None:
        vendor = nested_dict(merged, "vendor_or_sender")
        vendor["name"] = updates["vendor_name"]
        merged["vendor_or_sender"] = vendor
    if updates.get("gstin") is not None:
        identifiers = nested_dict(merged, "identifiers")
        identifiers["gstin"] = updates["gstin"]
        merged["identifiers"] = identifiers
    if updates.get("invoice_date") is not None:
        dates = nested_dict(merged, "dates")
        dates["invoice_date"] = updates["invoice_date"]
        merged["dates"] = dates
    if updates.get("invoice_number") is not None:
        identifiers = nested_dict(merged, "identifiers")
        identifiers["invoice_number"] = updates["invoice_number"]
        merged["identifiers"] = identifiers
    if updates.get("total_amount") is not None or updates.get("currency") is not None:
        amounts = nested_dict(merged, "amounts")
        if updates.get("total_amount") is not None:
            amounts["total"] = updates["total_amount"]
        if updates.get("currency") is not None:
            amounts["currency"] = updates["currency"]
        merged["amounts"] = amounts
    if updates.get("gst_amount") is not None:
        amounts = nested_dict(merged, "amounts")
        amounts["gst"] = updates["gst_amount"]
        merged["amounts"] = amounts
    if updates.get("category") is not None:
        merged["expense_category"] = updates["category"]
    if updates.get("notes") is not None:
        merged["notes"] = updates["notes"]
    return merged
