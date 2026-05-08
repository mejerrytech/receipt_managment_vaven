import json
from typing import Any, Dict, List

from shared.openai_client import OpenAIService

openai_service = OpenAIService()


def _safe_json_loads(extracted_json: str) -> dict:
    try:
        parsed = json.loads(extracted_json)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _compact_ocr_payload_for_summary(extracted_json: str) -> Dict[str, Any]:
    data = _safe_json_loads(extracted_json)
    data.pop("tables", None)
    data.pop("confidence", None)
    data.pop("_ocr_metadata", None)

    amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
    vendor_block = data.get("vendor_or_sender") if isinstance(data.get("vendor_or_sender"), dict) else {}
    identifiers = data.get("identifiers") if isinstance(data.get("identifiers"), dict) else {}
    items = data.get("items") if isinstance(data.get("items"), list) else []

    fields: Dict[str, Any] = {
        "document_type": data.get("document_type"),
        "title": data.get("title"),
        "date": data.get("date") or data.get("document_date"),
        "vendor": data.get("vendor_name") or data.get("vendor") or vendor_block.get("name"),
        "vendor_address": vendor_block.get("address") or data.get("address"),
        "invoice_number": data.get("invoice_number") or identifiers.get("invoice_number"),
        "gstin": data.get("gstin") or identifiers.get("gstin") or vendor_block.get("gstin"),
        "currency": data.get("currency") or amounts.get("currency"),
        "total_amount": data.get("total_amount") or amounts.get("total"),
        "subtotal": amounts.get("subtotal") or data.get("subtotal"),
        "tax_amount": amounts.get("tax") or data.get("tax"),
        "cgst": amounts.get("cgst") or data.get("cgst"),
        "sgst": amounts.get("sgst") or data.get("sgst"),
        "igst": amounts.get("igst") or data.get("igst"),
    }
    compact_fields = {k: v for k, v in fields.items() if v not in (None, "", [], {})}

    compact_items: List[Dict[str, Any]] = []
    for item in items[:15]:
        if not isinstance(item, dict):
            continue
        clean_item = {
            "description": item.get("description") or item.get("name"),
            "quantity": item.get("quantity"),
            "unit_price": item.get("price"),
            "amount": item.get("amount") or item.get("total"),
        }
        clean_item = {k: v for k, v in clean_item.items() if v not in (None, "", [], {})}
        if clean_item:
            compact_items.append(clean_item)
    return {"fields": compact_fields, "items": compact_items}


async def build_upload_preview_card(extracted_json: str, confidence: float) -> str:
    """Build compact OCR summary card body with confidence border."""
    is_high_conf = confidence > 0.8
    compact_payload = _compact_ocr_payload_for_summary(extracted_json)
    payload_json = json.dumps(compact_payload, ensure_ascii=False, indent=2)

    system_prompt = """You create concise OCR summary cards for Telegram in Markdown.
Rules:
1. Use only provided data. Never invent missing values.
2. Skip missing fields entirely (do not print N/A).
3. If items exist, include bullet points with qty/unit/amount only when present.
4. Do not include labels: Image OCR Summary Card, Confidence, File, Unordered List.
5. Do not include JSON/code blocks."""
    user_prompt = f"Generate only the card body.\n\nData:\n{payload_json}"

    llm_summary = await openai_service.ask(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        use_memory=False
    )
    llm_summary = (llm_summary or "").strip()
    if not llm_summary:
        fields = compact_payload.get("fields", {})
        items = compact_payload.get("items", [])
        lines: List[str] = [f"**{k.replace('_', ' ').title()}:** {v}" for k, v in fields.items()]
        for idx, item in enumerate(items, 1):
            parts = [str(item.get("description", f"Item {idx}"))]
            if "quantity" in item:
                parts.append(f"qty: {item['quantity']}")
            if "unit_price" in item:
                parts.append(f"unit: {item['unit_price']}")
            if "amount" in item:
                parts.append(f"amount: {item['amount']}")
            lines.append(f"• {' | '.join(parts)}")
        llm_summary = "\n".join(lines) if lines else "_No extractable fields found._"

    border = "🟢────────────────────────" if is_high_conf else "🟠────────────────────────"
    return f"{border}\n{llm_summary}\n{border}"
