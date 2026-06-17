"""
OCR Service — thin compatibility shim.

All image validation and OCR extraction is now handled exclusively by
GeminiOCRService (gemini-2.5-flash).

The public symbols (OCRService, get_ocr_service) are preserved so that every
existing caller (bot handlers, Celery tasks, tests) continues to work without
any import changes.
"""

from shared.gemini_ocr_service import GeminiOCRService, get_gemini_ocr_service

# ── Public aliases (backwards-compatible) ─────────────────────────────────────
OCRService = GeminiOCRService


def get_ocr_service() -> GeminiOCRService:
    """Return the global OCR service instance (now Gemini-backed)."""
    return get_gemini_ocr_service()
