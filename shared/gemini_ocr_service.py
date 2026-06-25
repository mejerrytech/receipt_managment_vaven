"""
Gemini OCR Service — Single-model, single-call image validation + extraction.

Flow (one API call):
1. Gemini validates image quality.
   - If unreadable/blurry/low-quality → returns { "status": "unreadable", ... }
   - If readable → returns full structured extraction in the same response.
2. Return JSON string compatible with the existing pipeline.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

from shared.database import DatabaseService
from shared.llm_usage import record_gemini_response

load_dotenv()

logger = logging.getLogger("gemini_ocr_service")

GEMINI_MODEL = "gemini-2.5-flash"

COMBINED_PROMPT_TEMPLATE = """You are an expert OCR and document understanding assistant.

First, assess whether this image is readable:
- If it is blurry, too dark, too bright, heavily cropped, distorted, or the text simply
  cannot be read confidently, respond with exactly:
  {{"status": "unreadable", "message": "I couldn't clearly understand the uploaded image. Please re-upload a clearer image."}}

If the image IS readable, extract structured data from the document into JSON.
Do NOT generate markdown cards, prose summaries, or duplicate the same data in multiple fields.

Required JSON shape when readable (keep FLAT and compact):
- "status": "readable"
- "document_type", "title", "vendor_name", "document_date", "currency"
- "total_amount", "subtotal", "cgst", "sgst", "igst" (numbers only at top level)
- "invoice_number", "gstin", "vendor_address" when present
- "items": array with EVERY visible line item (do not skip, group, or summarise rows).
  Each item MUST use these keys: "description", "quantity", "unit_price", "amount"
  (do not use Particulars, Qty/Kg, N/Rate, or other column headers as keys).
- "expense_category": single best match from: {expense_categories}
- "confidence": {{"overall": 0.0-1.0}}

Rules:
- Include only fields actually visible on the document.
- Do NOT use nested tax objects, tax_slabs arrays, or duplicate summaries.
- Omit text_content unless absolutely necessary.
- Keep JSON compact — no display_card, no markdown, no commentary.
- Finish the JSON object completely; never stop mid-key or mid-array.
- If the user provided context about the upload, use it as an extra hint: {user_input_text}

Return ONLY the raw JSON object. Do not wrap it in markdown, do not add backticks,
do not add any explanation before or after the JSON.
"""

COMPACT_RETRY_PROMPT = """The previous OCR JSON was truncated or invalid.
Return ONE compact readable receipt JSON only.

Hard limits:
- Flat keys only (no nested tax_details / tax_slabs).
- Include EVERY line item visible on the receipt in "items".
- Include: status, document_type, vendor_name, document_date, currency, total_amount,
  subtotal, cgst, sgst, igst, invoice_number, gstin, expense_category, confidence.
- expense_category must be one of: {expense_categories}
- Complete valid JSON only. No markdown fences.
"""


class GeminiOCRService:
    """
    Single-model, single-call OCR service using Gemini.
    Public interface is identical to the old OCRService — no caller changes needed.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
                "Add it to .env — see .env.example for instructions."
            )
        self._client = genai.Client(api_key=api_key)
        self._expense_categories = DatabaseService.EXPENSE_CATEGORIES
        logger.info("GeminiOCRService initialized with model=%s", GEMINI_MODEL)

    def _image_part(self, image_bytes: bytes, mime_type: str) -> genai_types.Part:
        return genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Extract JSON from the model response.
        Tries clean parse first, then strips fences, then finds first { to last }.
        No regex on the content body — avoids truncation/performance issues.
        """
        def _strip_fences(text: str) -> str:
            t = (text or "").strip()
            if not t:
                return t
            if t.startswith("```"):
                newline = t.find("\n")
                t = t[newline + 1:] if newline != -1 else t[3:]
            if t.endswith("```"):
                t = t[:-3]
            return t.strip()

        def _sanitize_json_like(s: str) -> str:
            """
            Repair a few common model JSON issues WITHOUT regex:
            - Missing value after ':' -> insert null
            - Trailing comma before '}' or ']' -> remove
            This is conservative and does not invent keys.
            """
            if not s:
                return s

            out: list[str] = []
            i = 0
            in_str = False
            escape = False

            while i < len(s):
                ch = s[i]

                if in_str:
                    out.append(ch)
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    i += 1
                    continue

                if ch == '"':
                    in_str = True
                    out.append(ch)
                    i += 1
                    continue

                # Remove trailing commas before } or ]
                if ch == ",":
                    j = i + 1
                    while j < len(s) and s[j] in " \t\r\n":
                        j += 1
                    if j < len(s) and s[j] in "}]":
                        i += 1
                        continue
                    out.append(ch)
                    i += 1
                    continue

                # Insert null for missing values after ':'
                if ch == ":":
                    out.append(ch)
                    j = i + 1
                    while j < len(s) and s[j] in " \t\r\n":
                        out.append(s[j])
                        j += 1
                    if j < len(s) and s[j] in ",}]":
                        out.append("null")
                        i = j
                        continue
                    i += 1
                    continue

                out.append(ch)
                i += 1

            return "".join(out)

        text = (raw or "").strip()
        if not text:
            raise json.JSONDecodeError("Empty model response", "", 0)

        # 1. Clean parse (model followed instructions perfectly)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip opening fence (```json or ```) and closing fence
        text = _strip_fences(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3. Slice from first { to last } — no regex, just index arithmetic
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            sliced = text[start:end]
            # 3a) Try strict parse of the slice
            try:
                return json.loads(sliced)
            except json.JSONDecodeError:
                pass

            # 3b) Try to parse the first JSON object only (ignore trailing junk)
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(sliced)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

            # 3c) Repair a few common issues and retry (no regex)
            repaired = _sanitize_json_like(sliced)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(repaired)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        salvaged = self._salvage_truncated_json(text)
        if salvaged is not None:
            logger.warning("Recovered OCR JSON from truncated Gemini response")
            return salvaged

        raise json.JSONDecodeError("No valid JSON found in model response", text, 0)

    def _salvage_truncated_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """Best-effort recovery when the model stops mid-JSON."""
        text = (raw or "").strip()
        if text.startswith("```"):
            newline = text.find("\n")
            text = text[newline + 1:] if newline != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start = text.find("{")
        if start == -1:
            return None
        sliced = text[start:]

        for _ in range(min(len(sliced), 400)):
            candidate = sliced.rstrip()
            while candidate and candidate[-1] not in '}]0123456789}"':
                candidate = candidate[:-1]
            candidate = candidate.rstrip(",:")
            open_brackets = candidate.count("[") - candidate.count("]")
            open_braces = candidate.count("{") - candidate.count("}")
            if open_brackets < 0 or open_braces < 0:
                sliced = sliced[:-1]
                continue
            closed = candidate + ("]" * open_brackets) + ("}" * open_braces)
            try:
                obj = json.loads(closed)
            except json.JSONDecodeError:
                sliced = sliced[:-1]
                continue
            if not isinstance(obj, dict):
                sliced = sliced[:-1]
                continue
            if obj.get("status") == "unreadable":
                return obj
            if obj.get("total_amount") is not None or obj.get("items") or obj.get("line_items"):
                return obj
            sliced = sliced[:-1]
        return None

    def _ocr_config(self, *, call_type: str) -> genai_types.GenerateContentConfig:
        # thinking_budget=0 keeps output tokens for actual JSON instead of hidden reasoning.
        return genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=12288,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        )

    async def _call_gemini(self, contents: list, *, call_type: str = "ocr_extract") -> str:
        """Run the synchronous Gemini SDK call off the event loop."""
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=self._ocr_config(call_type=call_type),
        )
        record_gemini_response(
            response,
            model=GEMINI_MODEL,
            call_type=call_type,
            details={"parts": len(contents)},
        )

        text = getattr(response, "text", None)
        if text:
            return text

        # response.text can be None when the model is truncated or blocked.
        # Try to recover the text from candidate parts, otherwise log why.
        finish_reason = None
        try:
            candidate = (response.candidates or [None])[0]
            if candidate is not None:
                finish_reason = getattr(candidate, "finish_reason", None)
                parts = getattr(getattr(candidate, "content", None), "parts", None) or []
                recovered = "".join(getattr(p, "text", "") or "" for p in parts)
                if recovered.strip():
                    return recovered
        except Exception:
            logger.exception("Failed to recover text from Gemini candidates")

        logger.error(
            "Gemini returned no text (finish_reason=%s). Likely truncated by token limit.",
            finish_reason,
        )
        return ""

    async def extract_data(
        self,
        image_bytes: bytes,
        mime_type: str,
        custom_prompt: Optional[str] = None,  # kept for API compatibility
        user_input_text: Optional[str] = None,
    ) -> str:
        """
        Validate image quality and extract structured data in a single Gemini call.

        Returns a JSON string:
          Unreadable: { "status": "unreadable", "message": "...", "_ocr_metadata": {...} }
          Success:    { <document fields>, "_ocr_metadata": {...} }
        """
        logger.info(
            "GeminiOCRService.extract_data called mime_type=%s user_input=%s",
            mime_type,
            bool(user_input_text),
        )

        categories_str = ", ".join(self._expense_categories)
        prompt = COMBINED_PROMPT_TEMPLATE.format(
            expense_categories=categories_str,
            user_input_text=f'"{user_input_text}"' if user_input_text else "null",
        )

        raw: str = ""
        image_part = self._image_part(image_bytes, mime_type)
        try:
            raw = await self._call_gemini([prompt, image_part])
            result: Dict[str, Any] = self._parse_json(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Gemini OCR JSON parse failed; retrying compact extraction. First 400 chars: %s",
                raw[:400],
            )
            retry_prompt = COMPACT_RETRY_PROMPT.format(
                expense_categories=categories_str,
            )
            try:
                raw = await self._call_gemini(
                    [retry_prompt, image_part],
                    call_type="ocr_extract_retry",
                )
                result = self._parse_json(raw)
                logger.info("Gemini compact OCR retry succeeded")
            except json.JSONDecodeError:
                logger.error(
                    "Gemini OCR retry still invalid. First 400 chars: %s", raw[:400]
                )
                return json.dumps({
                    "status": "unreadable",
                    "message": (
                        "I could read part of the receipt but could not finish extraction. "
                        "Please try again with a clearer, flatter photo."
                    ),
                    "_ocr_metadata": {
                        "model_used": GEMINI_MODEL,
                        "overall_confidence": 0.0,
                        "status": "parse_error",
                        "fallback_used": True,
                    },
                })
        except Exception:
            logger.exception("Gemini API call failed")
            raise

        # Unreadable path
        if result.get("status") == "unreadable":
            logger.warning("Gemini marked image as unreadable: %s", result.get("message"))
            return json.dumps({
                "status": "unreadable",
                "message": result.get(
                    "message",
                    "I couldn't clearly understand the uploaded image. Please re-upload a clearer image.",
                ),
                "_ocr_metadata": {
                    "model_used": GEMINI_MODEL,
                    "overall_confidence": 0.0,
                    "status": "unreadable",
                    "fallback_used": False,
                },
            })

        # Readable path
        result.pop("status", None)

        confidence_block = result.get("confidence", {})
        if not isinstance(confidence_block, dict):
            confidence_block = {}
            result["confidence"] = confidence_block

        overall_confidence = float(confidence_block.get("overall", 0.65))

        logger.info(
            "Gemini OCR extraction successful overall_confidence=%.2f fields=%s items=%s",
            overall_confidence,
            len([k for k in result if k not in ("confidence", "text_content", "status")]),
            len(result.get("items") or result.get("line_items") or []),
        )

        return json.dumps({
            **result,
            "_ocr_metadata": {
                "model_used": GEMINI_MODEL,
                "overall_confidence": overall_confidence,
                "field_confidences": confidence_block,
                "status": "confirmed" if overall_confidence >= 0.80 else "pending_review",
                "fallback_used": False,
            },
        })


# ─── Singleton ────────────────────────────────────────────────────────────────

_gemini_ocr_service: Optional[GeminiOCRService] = None


def get_gemini_ocr_service() -> GeminiOCRService:
    global _gemini_ocr_service
    if _gemini_ocr_service is None:
        _gemini_ocr_service = GeminiOCRService()
    return _gemini_ocr_service
