import mimetypes
import os
from uuid import UUID
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from api.review.constants import IMAGE_DIR
from shared.database import PendingDocument


def ensure_image_dir() -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_DIR


def ext_for_mime(mime_type: Optional[str]) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
    }
    if mime_type in mapping:
        return mapping[mime_type]
    return mimetypes.guess_extension(mime_type or "") or ".bin"


def image_path(pending_id: UUID, mime_type: Optional[str] = None) -> Path:
    pid = str(pending_id)
    base = ensure_image_dir() / pid
    if mime_type:
        return base.with_suffix(ext_for_mime(mime_type))
    for path in base.parent.glob(f"{pid}.*"):
        return path
    return base.with_suffix(".jpg")


def save_review_image(pending_id: UUID, file_bytes: bytes, mime_type: str) -> Path:
    path = image_path(pending_id, mime_type)
    path.write_bytes(file_bytes)
    return path


def find_review_image(pending_id: UUID) -> Optional[Path]:
    pid = str(pending_id)
    base = ensure_image_dir() / pid
    matches = list(base.parent.glob(f"{pid}.*"))
    return matches[0] if matches else None


def image_url(pending_id: UUID, user_id: UUID) -> str:
    return f"/api/review/receipts/{pending_id}/image?user_id={user_id}"


async def download_telegram_image(file_id: str) -> bytes:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Telegram bot token not configured")
    try:
        from telegram import Bot
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="python-telegram-bot not installed") from exc

    bot = Bot(token=token)
    tg_file = await bot.get_file(file_id)
    return bytes(await tg_file.download_as_bytearray())


async def resolve_image_bytes(pending: PendingDocument) -> tuple[bytes, str]:
    local = find_review_image(pending.id)
    if local and local.is_file():
        mime = pending.mime_type or mimetypes.guess_type(local.name)[0] or "image/jpeg"
        return local.read_bytes(), mime

    if pending.telegram_file_id:
        file_bytes = await download_telegram_image(pending.telegram_file_id)
        mime = pending.mime_type or "image/jpeg"
        save_review_image(pending.id, file_bytes, mime)
        return file_bytes, mime

    raise HTTPException(status_code=404, detail="Receipt image not found")
