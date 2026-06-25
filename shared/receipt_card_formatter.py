"""Deterministic receipt summary cards from OCR JSON (no LLM)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

WHATSAPP_PART_LIMIT = 1590  # Twilio hard limit is 1600; leave a small safety margin
_SEPARATOR = "━━━━━━━━━━━━━━━━"
_COMPACT_SEPARATOR = "────────────"


def _safe_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _format_money(value: Any, currency: str = "INR", *, compact: bool = False) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        amount = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return str(value)
    symbol = "₹" if (currency or "INR").upper() in ("INR", "RS", "RUPEES") else f"{currency} "
    if compact and amount == int(amount):
        return f"{symbol}{int(amount):,}"
    return f"{symbol}{amount:,.2f}"


def _get_field(raw: Dict[str, Any], *keys: str) -> Any:
    """Case-insensitive field lookup (Indian invoices use Particular, Qty/Kg, Rate, Value)."""
    lower_map = {str(k).lower(): v for k, v in raw.items()}
    for key in keys:
        if key in raw and raw[key] not in (None, "", [], {}):
            return raw[key]
        hit = lower_map.get(key.lower())
        if hit not in (None, "", [], {}):
            return hit
    return None


def _field_by_key_hint(raw: Dict[str, Any], hints: tuple[str, ...]) -> Any:
    """Match fields like Particulars, N/Rate, Qty/Kg when exact keys differ."""
    for key, value in raw.items():
        if value in (None, "", [], {}):
            continue
        normalized = str(key).lower().replace(" ", "").replace("_", "")
        for hint in hints:
            if hint in normalized:
                return value
    return None


def _normalize_item(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    desc = _get_field(
        raw,
        "description",
        "name",
        "product_name",
        "item_name",
        "title",
        "particular",
        "particulars",
        "item",
        "product",
    ) or _field_by_key_hint(raw, ("particular", "description", "productname", "itemname", "product", "item"))
    qty = _get_field(raw, "quantity", "qty", "qty/kg", "qty_kg", "qty per kg") or _field_by_key_hint(
        raw, ("qty", "quantity")
    )
    unit_price = _get_field(
        raw, "unit_price", "price", "rate", "mrp", "n/rate", "n_rate", "netrate", "net_rate"
    ) or _field_by_key_hint(raw, ("rate", "mrp", "unitprice", "price"))
    amount = _get_field(raw, "amount", "total", "line_total", "value", "net_amount") or _field_by_key_hint(
        raw, ("value", "amount", "linetotal", "netamount")
    )
    if not any(v not in (None, "", [], {}) for v in (desc, qty, unit_price, amount)):
        return None
    clean = {
        "description": str(desc).strip() if desc else "Item",
        "quantity": qty,
        "unit_price": unit_price,
        "amount": amount,
    }
    return {k: v for k, v in clean.items() if v not in (None, "", [], {})}


def _collect_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect every line item the OCR JSON provides (no artificial cap)."""
    seen: List[Dict[str, Any]] = []

    def _append(raw_items: Any) -> None:
        if not isinstance(raw_items, list):
            return
        for raw in raw_items:
            item = _normalize_item(raw)
            if item:
                seen.append(item)

    for key in ("items", "line_items", "products", "entries", "purchases"):
        _append(data.get(key))

    tax_details = data.get("tax_details")
    if isinstance(tax_details, dict):
        _append(tax_details.get("tax_slabs"))
        _append(tax_details.get("items"))

    tables = data.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if isinstance(table, dict):
                _append(table.get("rows"))
                _append(table.get("items"))

    return seen


def compact_ocr_payload(extracted_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    data = _safe_dict(extracted_json)
    for key in ("tables", "confidence", "_ocr_metadata", "display_card", "status", "message"):
        data.pop(key, None)

    amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
    vendor_block = data.get("vendor_or_sender") if isinstance(data.get("vendor_or_sender"), dict) else {}
    identifiers = data.get("identifiers") if isinstance(data.get("identifiers"), dict) else {}
    tax_details = data.get("tax_details") if isinstance(data.get("tax_details"), dict) else {}

    currency = data.get("currency") or amounts.get("currency") or "INR"

    fields: Dict[str, Any] = {
        "document_type": data.get("document_type"),
        "title": data.get("title"),
        "date": data.get("date") or data.get("document_date"),
        "vendor": data.get("vendor_name") or data.get("vendor") or vendor_block.get("name"),
        "vendor_address": vendor_block.get("address") or data.get("vendor_address") or data.get("address"),
        "invoice_number": data.get("invoice_number") or identifiers.get("invoice_number"),
        "gstin": data.get("gstin") or identifiers.get("gstin") or vendor_block.get("gstin"),
        "fssai": data.get("fssai") or identifiers.get("fssai"),
        "phone": data.get("phone") or vendor_block.get("phone") or data.get("contact"),
        "currency": currency,
        "total_amount": data.get("total_amount") or amounts.get("total"),
        "subtotal": amounts.get("subtotal") or data.get("subtotal") or tax_details.get("taxable_amount_total"),
        "tax_amount": amounts.get("tax") or data.get("tax"),
        "cgst": amounts.get("cgst") or data.get("cgst") or tax_details.get("cgst_total"),
        "sgst": amounts.get("sgst") or data.get("sgst") or tax_details.get("sgst_total"),
        "igst": amounts.get("igst") or data.get("igst") or tax_details.get("igst_total"),
        "expense_category": data.get("expense_category"),
    }
    compact_fields = {k: v for k, v in fields.items() if v not in (None, "", [], {})}
    compact_items = _collect_items(data)

    confidence = data.get("confidence")
    overall_confidence = None
    if isinstance(confidence, dict):
        overall_confidence = confidence.get("overall")
    elif confidence is not None:
        overall_confidence = confidence

    return {
        "fields": compact_fields,
        "items": compact_items,
        "overall_confidence": overall_confidence,
    }


def _confidence_label(score: Optional[Any]) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "Review"
    if value >= 0.85:
        return f"High ({value:.0%})"
    if value >= 0.65:
        return f"Medium ({value:.0%})"
    return f"Low ({value:.0%})"


def _shorten(text: str, max_len: Optional[int]) -> str:
    text = (text or "").strip()
    if not max_len or len(text) <= max_len:
        return text
    if max_len <= 1:
        return text[:max_len]
    return text[: max_len - 1].rstrip() + "…"


def _format_item_line(
    index: int,
    item: Dict[str, Any],
    currency: str,
    *,
    style: str = "full",
    max_desc_len: Optional[int] = None,
) -> str:
    desc = _shorten(str(item.get("description") or f"Item {index}"), max_desc_len)
    qty = item.get("quantity")
    unit_price = item.get("unit_price")
    amount = item.get("amount")
    money_compact = style == "minimal"

    if style in ("compact", "minimal"):
        if qty is not None and unit_price is not None and amount is not None:
            return (
                f"{index}. {desc} "
                f"{qty}×{_format_money(unit_price, currency, compact=money_compact)}"
                f"={_format_money(amount, currency, compact=money_compact)}"
            )
        if amount is not None:
            return f"{index}. {desc} — {_format_money(amount, currency, compact=money_compact)}"
        if qty is not None:
            return f"{index}. {desc} (qty {qty})"
        return f"{index}. {desc}"

    left = f"{index}. {desc}"
    if qty is not None and unit_price is not None:
        detail = f"   {qty} x {_format_money(unit_price, currency)}"
        if amount is not None:
            detail += f" = {_format_money(amount, currency)}"
        return f"{left}\n{detail}"
    if amount is not None:
        return f"{left} — {_format_money(amount, currency)}"
    if qty is not None:
        return f"{left} — qty {qty}"
    return left


def format_receipt_card_markdown(
    extracted_json: Union[str, Dict[str, Any]],
    *,
    for_whatsapp: bool = False,
    style: str = "full",
    max_desc_len: Optional[int] = None,
) -> str:
    """Rich receipt card. Uses WhatsApp *bold* when for_whatsapp=True.

    style: full (default), compact (single-line items), or minimal (dense one-card layout).
    """
    payload = compact_ocr_payload(extracted_json)
    fields = payload.get("fields") or {}
    items: List[Dict[str, Any]] = payload.get("items") or []
    currency = str(fields.get("currency") or "INR")

    if not fields and not items:
        return ""

    b = (lambda text: f"*{text}*") if for_whatsapp else (lambda text: f"**{text}**")
    separator = _COMPACT_SEPARATOR if style in ("compact", "minimal") else _SEPARATOR

    heading = fields.get("document_type") or fields.get("title") or "Receipt Summary"
    lines: List[str] = []

    if style == "minimal":
        header_bits = [str(heading)]
        if fields.get("vendor"):
            header_bits.append(str(fields["vendor"]))
        if fields.get("date"):
            header_bits.append(str(fields["date"]))
        lines.append("🧾 " + b(" · ".join(header_bits)))
        if fields.get("invoice_number"):
            lines.append(f"Inv: {fields['invoice_number']}")
        if fields.get("gstin"):
            lines.append(f"GSTIN: {fields['gstin']}")
    else:
        lines.extend(["🧾 " + b(str(heading)), separator])
        meta = [
            ("vendor", "🏪 Vendor", fields.get("vendor")),
            ("date", "📅 Date", fields.get("date")),
            ("invoice_number", "🧾 Invoice", fields.get("invoice_number")),
            ("gstin", "🆔 GSTIN", fields.get("gstin")),
            ("fssai", "📋 FSSAI", fields.get("fssai")),
            ("vendor_address", "📍 Address", fields.get("vendor_address")),
            ("phone", "📞 Contact", fields.get("phone")),
        ]
        for _key, label, value in meta:
            if value not in (None, "", [], {}):
                if style == "compact" and _key == "vendor_address" and len(str(value)) > 60:
                    value = _shorten(str(value), 60)
                lines.append(f"{label}: {value}")

    if items:
        if style == "minimal":
            lines.append(b(f"ITEMS ({len(items)})"))
        else:
            lines.extend(["", b(f"ITEMS ({len(items)})"), separator])
        for idx, item in enumerate(items, 1):
            lines.append(_format_item_line(idx, item, currency, style=style, max_desc_len=max_desc_len))

    total_lines: List[str] = []
    money_fields = [
        ("subtotal", "Subtotal", fields.get("subtotal")),
        ("cgst", "CGST", fields.get("cgst")),
        ("sgst", "SGST", fields.get("sgst")),
        ("igst", "IGST", fields.get("igst")),
        ("tax_amount", "Tax", fields.get("tax_amount")),
        ("total_amount", "Grand Total", fields.get("total_amount")),
    ]
    money_compact = style == "minimal"
    for _key, label, value in money_fields:
        if value not in (None, "", [], {}):
            formatted = _format_money(value, currency, compact=money_compact)
            if label == "Grand Total":
                total_lines.append(b(f"{label}: {formatted}"))
            else:
                total_lines.append(f"{label}: {formatted}")

    if total_lines:
        if style == "minimal":
            lines.append(" | ".join(total_lines))
        else:
            lines.extend(["", b("TOTALS"), separator, *total_lines])

    if fields.get("expense_category"):
        lines.append(f"🏷️ {fields['expense_category']}")

    if style != "minimal":
        conf = payload.get("overall_confidence")
        if conf is not None:
            emoji = "🟢" if float(conf) >= 0.85 else "🟠" if float(conf) >= 0.65 else "🔴"
            lines.append(f"{emoji} Confidence: {_confidence_label(conf)}")

    return "\n".join(lines).strip()


def _whatsapp_footer(pending_id: int) -> str:
    return (
        f"\n—\n"
        f"Pending ID: {pending_id}\n"
        f"CONFIRM {pending_id} to save\n"
        f"EDIT {pending_id} <json>"
    )


def _fit_whatsapp_card_body(
    data: Dict[str, Any],
    budget: int,
) -> str:
    """Pick the richest layout that still fits in one WhatsApp bubble."""
    for style in ("full", "compact", "minimal"):
        body = format_receipt_card_markdown(data, for_whatsapp=True, style=style)
        if len(body) <= budget:
            return body

    for max_desc in (40, 32, 24, 18, 14, 10):
        body = format_receipt_card_markdown(
            data, for_whatsapp=True, style="minimal", max_desc_len=max_desc
        )
        if len(body) <= budget:
            return body

    body = format_receipt_card_markdown(
        data, for_whatsapp=True, style="minimal", max_desc_len=8
    )
    return truncate_card_body(body, budget)


def build_whatsapp_review_messages(
    extracted_json: Union[str, Dict[str, Any]],
    pending_id: int,
    *,
    max_chars: int = WHATSAPP_PART_LIMIT,
) -> List[str]:
    """Build exactly one WhatsApp review card (never split into multiple bubbles)."""
    data = _safe_dict(extracted_json)
    footer = _whatsapp_footer(pending_id)
    budget = max_chars - len(footer)
    if budget < 200:
        budget = max_chars - 120

    body = _fit_whatsapp_card_body(data, budget)
    if not body:
        return [append_whatsapp_review_footer("_Could not generate review card._", pending_id)]

    message = f"{body}{footer}"
    if len(message) > max_chars:
        body = truncate_card_body(body, budget)
        message = f"{body}{footer}"
    return [message]


def truncate_card_body(body: str, max_chars: int) -> str:
    text = (body or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def append_whatsapp_review_footer(body: str, pending_id: int) -> str:
    return f"{(body or '').strip()}{_whatsapp_footer(pending_id)}"


def build_whatsapp_review_from_ocr(
    extracted_json: Union[str, Dict[str, Any]],
    pending_id: int,
    *,
    max_chars: int = WHATSAPP_PART_LIMIT,
) -> str:
    """Single WhatsApp review message."""
    return build_whatsapp_review_messages(extracted_json, pending_id, max_chars=max_chars)[0]
