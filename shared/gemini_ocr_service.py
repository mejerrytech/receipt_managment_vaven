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

load_dotenv()

logger = logging.getLogger("gemini_ocr_service")

GEMINI_MODEL = "gemini-2.5-flash"

COMBINED_PROMPT_TEMPLATE = """You are an expert OCR and document understanding assistant.

First, assess whether this image is readable:
- If it is blurry, too dark, too bright, heavily cropped, distorted, or the text simply
  cannot be read confidently, respond with exactly:
  {{"status": "unreadable", "message": "I couldn't clearly understand the uploaded image. Please re-upload a clearer image."}}

If the image IS readable, perform OCR on every visible character and then extract all
meaningful information from the document. Think of this as two tasks in one:
  1. Read everything you can see, word for word.
  2. Organise what you read into a clean JSON structure.

For the JSON output:
- Include any fields that are actually present in the document (vendor, date, amounts,
  line items, invoice/order numbers, tax IDs, addresses, recipient, etc.).
- For financial documents capture totals, subtotals, taxes and individual line items.
- Always include GST/tax fields when visible on the document:
  * "identifiers": {{ "gstin": "<15-char GST number>", "invoice_number": "..." }}
  * "amounts": {{ "total": <number>, "currency": "INR", "gst": <total tax if shown> }}
  * "taxes": {{ "cgst": <number or null>, "sgst": <number or null>, "igst": <number or null> }}
  Use CGST+SGST for intra-state bills and IGST for inter-state bills. Use null for missing components.
- Set "expense_category" to the single best match from: {expense_categories}
- Add a "confidence" object with an "overall" score (0.0-1.0) and per-field scores for
  the fields you extracted.
- Put the full raw OCR text in "text_content".
- Add a "display_card" string: a COMPLETE Markdown summary card for the user.
  * Start with a bold heading (e.g. **Tax Invoice** or **Receipt Summary**).
  * Show every extracted field as **Label:** value on its own line.
  * List EVERY line item as a bullet with description, qty, unit price, and amount.
  * Include all tax breakdowns and totals.
  * NO conversational intro ("This is a tax invoice from..."), NO prose paragraph.
  * Must include ALL items — never summarise or skip line items.
- Set "status" to "readable".
- If the user provided context about the upload, use it as an extra hint: {user_input_text}

Return ONLY the raw JSON object. Do not wrap it in markdown, do not add backticks,
do not add any explanation before or after the JSON.
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
        logger.info("GeminiOCRService initialized with model=%s", GEMINI_MODEL)

    @property
    def _expense_categories(self) -> list[str]:
        return DatabaseService.get_expense_categories_for_ai()

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

        raise json.JSONDecodeError("No valid JSON found in model response", text, 0)

    async def _call_gemini(self, contents: list) -> str:
        """Run the synchronous Gemini SDK call off the event loop."""
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=16384,
            ),
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
        try:
            raw = await self._call_gemini([prompt, self._image_part(image_bytes, mime_type)])
            result: Dict[str, Any] = self._parse_json(raw)
        except json.JSONDecodeError:
            logger.error(
                "Gemini response could not be parsed as JSON. First 400 chars: %s", raw[:400]
            )
            return json.dumps({
                "status": "unreadable",
                "message": (
                    "I couldn't clearly understand the uploaded image. "
                    "Please re-upload a clearer image."
                ),
                "_ocr_metadata": {
                    "model_used": GEMINI_MODEL,
                    "overall_confidence": 0.0,
                    "status": "parse_error",
                    "fallback_used": False,
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

        display_card = result.get("display_card")
        if isinstance(display_card, str) and display_card.strip():
            logger.info(
                "Gemini OCR extraction successful overall_confidence=%.2f display_card_chars=%s",
                overall_confidence,
                len(display_card.strip()),
            )
        else:
            logger.info("Gemini OCR extraction successful overall_confidence=%.2f (no display_card)", overall_confidence)

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
