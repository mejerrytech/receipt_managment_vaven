import asyncio
import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from shared.llm_usage import record_gemini_response

load_dotenv()

logger = logging.getLogger("upload_card_service")

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """You create complete OCR summary cards for messaging apps in Markdown.

Rules:
1. Use ONLY the provided OCR data. Never invent missing values.
2. Skip missing fields entirely — do not print N/A.
3. Output a structured Markdown CARD, NOT a prose paragraph or essay.
4. NEVER start with conversational phrases like "This is a tax invoice from...", "Here's a summary...", or similar filler.
5. Start with a bold heading (e.g. **Receipt Summary** or **Tax Invoice**).
6. Show each available field on its own line using **Label:** value markdown.
7. If line items exist, list EVERY item as a bullet point with description, qty, unit price, and amount when present. Never summarise, group, or skip items.
8. Include all tax breakdowns, subtotals, and totals when present in the data.
9. Do not include JSON, code blocks, confidence scores, or technical metadata.
10. The card must be COMPLETE — include all data from the input, not a shortened version."""

USER_PROMPT = """Generate only the card body from this OCR data:

{ocr_data}"""

RETRY_PROMPT = """Your previous answer was too short or written as prose.
Rewrite as a COMPLETE Markdown card:
- Bold heading first
- **Label:** value for every field in the data
- A bullet for EVERY line item (never skip or summarise items)
- All taxes and totals at the end
- NO conversational intro, NO paragraph essay

OCR Data:
{ocr_data}"""

# Twilio WhatsApp body hard limit is 1600; prompt targets a safe budget for one bubble.
WHATSAPP_MAX_CHARS = 1500

WHATSAPP_REVIEW_SYSTEM = f"""You create ONE complete WhatsApp review message for a receipt upload bot.

Hard rules:
1. TOTAL output MUST be at most {WHATSAPP_MAX_CHARS} characters including newlines. Count carefully.
2. Use ONLY the provided OCR JSON. Never invent missing values.
3. Include ALL important extracted data: vendor, GST number, Shop location, FSSAI number, Shop Address, Shop Contact number, date, every line item (compact bullets), taxes, totals.
4. Format as a Markdown card: bold heading, **Label:** value per line, bullet items.
5. End with exactly these action lines (use the given pending_id):
   —
   Pending ID: <pending_id>
   CONFIRM <pending_id> to save | EDIT <pending_id> <json>
6. NO filler prose, NO JSON blocks, NO "part 2" or "continued".
7. If space is tight, shorten labels/descriptions but keep every item row and all amounts accurate."""

WHATSAPP_REVIEW_USER = """OCR data:
{ocr_data}

pending_id: {pending_id}

Generate the complete single WhatsApp review message now (max {max_chars} chars)."""

WHATSAPP_REVIEW_RETRY = """Your previous message was {actual_chars} characters. WhatsApp rejects anything over {max_chars}.
Rewrite as ONE shorter message under {max_chars} characters. Keep every line item and all totals accurate.
Keep the CONFIRM/EDIT footer with pending_id {pending_id}.

Previous attempt:
{previous}

OCR data:
{ocr_data}"""

_INTERNAL_KEYS = frozenset({"_ocr_metadata", "confidence", "status", "tables", "display_card"})


def _safe_json_loads(extracted_json: str) -> dict:
    try:
        parsed = json.loads(extracted_json)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _strip_internal_fields(data: dict) -> dict:
    """Remove internal metadata only — keep the raw OCR structure as-is for the LLM."""
    return {k: v for k, v in data.items() if k not in _INTERNAL_KEYS and v not in (None, "", [], {})}


def _item_count(ocr_data: dict) -> int:
    for key in ("items", "line_items"):
        items = ocr_data.get(key)
        if isinstance(items, list):
            return len(items)
    return 0


def _looks_incomplete(summary: str, ocr_data: dict) -> bool:
    text = (summary or "").strip()
    if not text:
        return True
    lower = text.lower()
    if lower.startswith(("this is ", "here's ", "here is ", "you have ", "your ", "a tax invoice")):
        return True
    item_count = _item_count(ocr_data)
    bullet_count = text.count("•") + text.count("\n- ") + text.count("\n* ")
    if item_count > 2 and bullet_count < item_count:
        return True
    if item_count > 0 and len(text) < 500:
        return True
    return False


class _GeminiSummaryClient:
    """Gemini client for OCR card summarisation — always prompt-based."""

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set in .env")
        self._client = genai.Client(api_key=api_key)

    def _config(self, system_instruction: str = SYSTEM_INSTRUCTION) -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=8192,
        )

    async def _generate(self, prompt: str, *, system_instruction: str = SYSTEM_INSTRUCTION) -> Optional[str]:
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=GEMINI_MODEL,
                contents=[prompt],
                config=self._config(system_instruction),
            )
            record_gemini_response(
                response,
                model=GEMINI_MODEL,
                call_type="ocr_summary",
                details={"prompt_chars": len(prompt)},
            )
            return (response.text or "").strip() or None
        except Exception:
            logger.exception("Gemini summary call failed")
            return None

    async def whatsapp_review(self, ocr_data: dict, pending_id: int) -> Optional[str]:
        """Prompt-only: one WhatsApp review bubble with confirm/edit footer."""
        ocr_json = json.dumps(ocr_data, ensure_ascii=False, indent=2)
        message = await self._generate(
            WHATSAPP_REVIEW_USER.format(
                ocr_data=ocr_json,
                pending_id=pending_id,
                max_chars=WHATSAPP_MAX_CHARS,
            ),
            system_instruction=WHATSAPP_REVIEW_SYSTEM,
        )
        retries = 0
        while message and len(message) > WHATSAPP_MAX_CHARS and retries < 2:
            logger.warning(
                "WhatsApp review message too long chars=%s (max=%s), prompt retry %s",
                len(message),
                WHATSAPP_MAX_CHARS,
                retries + 1,
            )
            message = await self._generate(
                WHATSAPP_REVIEW_RETRY.format(
                    actual_chars=len(message),
                    max_chars=WHATSAPP_MAX_CHARS,
                    pending_id=pending_id,
                    previous=message,
                    ocr_data=ocr_json,
                ),
                system_instruction=WHATSAPP_REVIEW_SYSTEM,
            )
            retries += 1
        if message:
            logger.info("WhatsApp review message generated chars=%s pending_id=%s", len(message), pending_id)
        return message

    async def summarise(self, ocr_data: dict) -> Optional[str]:
        ocr_json = json.dumps(ocr_data, ensure_ascii=False, indent=2)
        summary = await self._generate(USER_PROMPT.format(ocr_data=ocr_json))
        if summary and _looks_incomplete(summary, ocr_data):
            logger.warning(
                "Gemini summary looked incomplete (chars=%s items=%s), retrying",
                len(summary),
                _item_count(ocr_data),
            )
            summary = await self._generate(RETRY_PROMPT.format(ocr_data=ocr_json))
        if summary:
            logger.info("Upload card summary generated chars=%s items=%s", len(summary), _item_count(ocr_data))
        return summary


_summary_client: Optional[_GeminiSummaryClient] = None


def _get_summary_client() -> _GeminiSummaryClient:
    global _summary_client
    if _summary_client is None:
        _summary_client = _GeminiSummaryClient()
    return _summary_client


async def build_upload_preview_card(extracted_json: str, confidence: float) -> str:
    """Build OCR preview card — prefer Gemini OCR display_card, else prompt-based summary."""
    is_high_conf = confidence > 0.8
    data = _safe_json_loads(extracted_json)

    display_card = data.get("display_card")
    if isinstance(display_card, str) and display_card.strip():
        summary = display_card.strip()
        logger.info("Using Gemini OCR display_card chars=%s", len(summary))
    else:
        clean_data = _strip_internal_fields(data)
        summary = await _get_summary_client().summarise(clean_data)
        if not summary:
            summary = "_Could not generate summary. Please review and confirm the upload._"

    border = "🟢────────────────────────" if is_high_conf else "🟠────────────────────────"
    return f"{border}\n{summary}\n{border}"


async def build_whatsapp_review_message(extracted_json: str, pending_id: int) -> str:
    """Prompt-only WhatsApp review card (single bubble, includes CONFIRM/EDIT footer)."""
    data = _safe_json_loads(extracted_json)
    clean_data = _strip_internal_fields(data)
    message = await _get_summary_client().whatsapp_review(clean_data, pending_id)
    if message:
        return message
    return (
        f"_Could not generate review card._\n\n"
        f"Pending ID: {pending_id}\n"
        f"CONFIRM {pending_id} to save | EDIT {pending_id} <json>"
    )
