import asyncio
import logging
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from shared.celery_app import (
    celery_app,
    CELERY_TASK_MAX_RETRIES,
    CELERY_TASK_RETRY_BACKOFF_SECONDS,
    OCR_TASK_SOFT_TIME_LIMIT,
    OCR_TASK_TIME_LIMIT,
)
from shared.database import DatabaseService
from shared.ocr_service import get_ocr_service
from shared.upload_card_service import build_upload_preview_card

load_dotenv()

logger = logging.getLogger("ocr_tasks")
db_service = DatabaseService()
ocr_service = get_ocr_service()


def _run(coro):
    """
    Run async code from Celery sync context safely.
    If an event loop is already running in this thread, execute the coroutine
    in a separate thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        return future.result()


async def _send_telegram_message_async(chat_id: int, text: str, pending_id: int) -> Optional[int]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return None

    bot = Bot(token=token)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm & Save", callback_data=f"confirm:{pending_id}")],
        [InlineKeyboardButton("✏️ Edit Data", callback_data=f"edit:{pending_id}")]
    ])
    msg = await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=keyboard,
        parse_mode="Markdown"
    )
    return msg.message_id


def _send_telegram_message(chat_id: int, text: str, pending_id: int) -> Optional[int]:
    try:
        return _run(_send_telegram_message_async(chat_id=chat_id, text=text, pending_id=pending_id))
    except Exception:
        logger.exception("Failed to send OCR completion message for pending_id=%s", pending_id)
        return None


def _send_telegram_ready(chat_id: int, pending_id: int, extracted_json: str, confidence: float) -> Optional[int]:
    async def _build_and_send():
        card_text = await build_upload_preview_card(extracted_json=extracted_json, confidence=confidence)
        return await _send_telegram_message_async(chat_id=chat_id, text=card_text, pending_id=pending_id)

    try:
        return _run(_build_and_send())
    except Exception:
        logger.exception("Failed to send OCR ready summary for pending_id=%s", pending_id)
        return None


def _send_telegram_failure(chat_id: int, error_message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return

    async def _send():
        bot = Bot(token=token)
        await bot.send_message(
            chat_id=chat_id,
            text=f"OCR processing failed after retries.\nReason: {error_message[:300]}"
        )

    try:
        _run(_send())
    except Exception:
        logger.exception("Failed to send OCR failure message")


def _send_telegram_info(chat_id: int, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return

    async def _send():
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text)

    try:
        _run(_send())
    except Exception:
        logger.exception("Failed to send OCR info message")


@celery_app.task(
    bind=True,
    max_retries=CELERY_TASK_MAX_RETRIES,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=CELERY_TASK_RETRY_BACKOFF_SECONDS,
    retry_jitter=True,
    soft_time_limit=OCR_TASK_SOFT_TIME_LIMIT,
    time_limit=OCR_TASK_TIME_LIMIT,
    name="shared.tasks.ocr_tasks.process_pending_ocr",
)
def process_pending_ocr(self, pending_id: int, user_id: int):
    """
    Process one pending OCR job:
    - Download source file from Telegram
    - Run OCR extraction
    - Mark pending as ready/failed
    - Notify user in Telegram
    """
    pending = db_service.get_pending_document_for_job(pending_id)
    if not pending:
        logger.warning("Pending document %s not found for OCR job", pending_id)
        return {"success": False, "reason": "pending_not_found"}

    # Idempotency guard
    if pending.status in ("ready", "confirmed", "cancelled"):
        logger.info("Skipping OCR job for pending_id=%s status=%s", pending_id, pending.status)
        return {"success": True, "reason": "already_processed"}

    if pending.user_id != user_id:
        logger.warning("User mismatch for pending_id=%s", pending_id)
        return {"success": False, "reason": "user_mismatch"}

    if not pending.telegram_file_id:
        error = "Missing telegram_file_id for queued OCR processing."
        db_service.mark_pending_ocr_failed(pending_id, error, retry_count=self.request.retries)
        _send_telegram_failure(pending.telegram_chat_id, error)
        return {"success": False, "reason": "missing_telegram_file_id"}

    async def _extract_from_telegram_file(file_id: str, mime_type: str) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        bot = Bot(token=token)
        tg_file = await bot.get_file(file_id)
        file_bytes = await tg_file.download_as_bytearray()
        return await ocr_service.extract_data(image_bytes=bytes(file_bytes), mime_type=mime_type or "image/jpeg")

    try:
        result = _run(_extract_from_telegram_file(pending.telegram_file_id, pending.mime_type))
        duplicate_after_ocr = db_service.find_duplicate_by_extracted_fingerprint(user_id, result)
        if duplicate_after_ocr:
            db_service.mark_pending_ocr_failed(
                pending_id,
                "Duplicate content detected after OCR fingerprint check.",
                retry_count=self.request.retries
            )
            _send_telegram_info(
                pending.telegram_chat_id,
                "Duplicate image/document detected after OCR. New save is skipped."
            )
            return {"success": False, "reason": "duplicate_after_ocr"}

        try:
            import json as _json
            parsed = _json.loads(result)
            c = parsed.get("confidence")
            if isinstance(c, dict):
                confidence = float(c.get("overall") or 0.0)
            elif c is not None:
                confidence = float(c)
            else:
                confidence = 0.0
        except Exception:
            confidence = 0.0

        db_service.mark_pending_ocr_ready(
            pending_id=pending_id,
            extracted_json=result,
            confidence_overall=confidence
        )
        msg_id = _send_telegram_ready(
            chat_id=pending.telegram_chat_id,
            pending_id=pending_id,
            extracted_json=result,
            confidence=confidence
        )
        if msg_id:
            db_service.set_pending_telegram_message_id(pending_id, msg_id)
        return {"success": True, "pending_id": pending_id}

    except Exception as e:
        logger.exception("OCR task failed for pending_id=%s", pending_id)
        # Manual terminal failure marking when retries exhausted
        if self.request.retries >= CELERY_TASK_MAX_RETRIES:
            db_service.mark_pending_ocr_failed(pending_id, str(e), retry_count=self.request.retries)
            _send_telegram_failure(pending.telegram_chat_id, str(e))
        raise
