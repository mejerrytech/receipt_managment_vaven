import hashlib
import base64
import hmac
import json
import logging
import re
import time
import asyncio
import uuid
from xml.sax.saxutils import escape
from urllib.parse import parse_qs

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response

from bot_telegram.config import settings as telegram_settings
from bot_whatsapp.config import normalize_whatsapp_number, settings
from shared.database import DatabaseService
from shared.image_hash_service import generate_image_hashes
from shared.nlp_sql_service_v2 import get_nlp_sql_service_v2
from shared.ocr_service import get_ocr_service
from shared.openai_client import OpenAIService
from shared.rag_service import get_rag_service
from shared.upload_card_service import build_whatsapp_review_message
from shared.whatsapp_request_state import (
    WhatsappRequestRef,
    claim_process_once,
    claim_send_once,
    cleanup_request_state,
    extract_basic_fields_from_ocr_json,
    get_redis_client,
    init_request_state,
    mark_completed,
    mark_error,
    patch_state,
)

logger = logging.getLogger("whatsapp_bot")

app = FastAPI(title="WhatsApp Receipt Bot")
openai_service = OpenAIService()
db_service = DatabaseService()
nlp_service_v2 = get_nlp_sql_service_v2()
rag_service = get_rag_service()
ocr_service = get_ocr_service()
_REDIS_CLIENT = None
_LOCAL_SEND_CLAIMS: dict[str, float] = {}

SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
_MESSAGE_RESPONSE_CACHE: dict[str, tuple[float, str]] = {}


async def _twilio_form(request: Request) -> dict[str, str]:
    raw = (await request.body()).decode("utf-8")
    parsed = parse_qs(raw, keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _public_request_url(request: Request) -> str:
    """Return the externally visible URL Twilio signed."""
    if settings.WHATSAPP_PUBLIC_BASE_URL:
        return f"{settings.WHATSAPP_PUBLIC_BASE_URL}{request.url.path}"
    return str(request.url)


def _is_valid_twilio_signature(request: Request, form: dict[str, str]) -> bool:
    """Validate Twilio webhook signature when enabled."""
    if not settings.WHATSAPP_VERIFY_TWILIO_SIGNATURE:
        return True
    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature or not settings.TWILIO_AUTH_TOKEN:
        return False
    base = _public_request_url(request)
    for key in sorted(form):
        base += key + form[key]
    digest = hmac.new(
        settings.TWILIO_AUTH_TOKEN.encode("utf-8"),
        base.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


def _xml_response(*messages: str) -> Response:
    # Twilio expects XML; use a conservative content-type for maximum compatibility.
    return Response(content=_xml_content(*messages), media_type="text/xml; charset=utf-8")


def _xml_content(*messages: str) -> str:
    parts = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<Response>"]
    for text in messages:
        if text:
            parts.append(f"<Message>{escape(text)}</Message>")
    parts.append("</Response>")
    return "".join(parts)


def _cacheable_xml_response(message_sid: str, *messages: str) -> Response:
    content = _xml_content(*messages)
    preview = " | ".join((message or "").replace("\n", " ")[:180] for message in messages) if messages else "(empty)"
    logger.info(
        "WhatsApp TwiML response sid=%s chars=%s preview=%s",
        message_sid or "-",
        len(content),
        preview,
    )
    if message_sid:
        _MESSAGE_RESPONSE_CACHE[message_sid] = (time.time(), content)
        _cleanup_message_cache()
    return Response(content=content, media_type="text/xml; charset=utf-8")


def _cached_response_for(message_sid: str) -> Response | None:
    if not message_sid:
        return None
    cached = _MESSAGE_RESPONSE_CACHE.get(message_sid)
    if not cached:
        return None
    created_at, content = cached
    if time.time() - created_at > settings.WHATSAPP_MESSAGE_CACHE_SECONDS:
        _MESSAGE_RESPONSE_CACHE.pop(message_sid, None)
        return None
    return Response(content=content, media_type="text/xml; charset=utf-8")


def _cleanup_message_cache() -> None:
    now = time.time()
    expired = [
        sid for sid, (created_at, _) in _MESSAGE_RESPONSE_CACHE.items()
        if now - created_at > settings.WHATSAPP_MESSAGE_CACHE_SECONDS
    ]
    for sid in expired:
        _MESSAGE_RESPONSE_CACHE.pop(sid, None)


# Twilio hard limit is 1600 chars per WhatsApp message body.
WHATSAPP_PART_LIMIT = 1500


def _split_for_whatsapp(text: str, max_length: int = WHATSAPP_PART_LIMIT) -> list[str]:
    """Split a long message into WhatsApp-sized parts on line boundaries.

    Twilio rejects any single message body over 1600 chars (error 21617), so a
    long OCR card MUST be delivered as several separate messages.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        # A single line longer than the limit must be hard-split.
        while len(line) > max_length:
            parts.append(line[:max_length])
            line = line[max_length:]
        current = line
    if current:
        parts.append(current)
    return parts


def _send_whatsapp_message_sync(to: str, body: str) -> None:
    """Send a WhatsApp reply via Twilio REST API, splitting into ≤1500-char parts."""
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        raise RuntimeError("Twilio credentials not configured")

    parts = _split_for_whatsapp(body)
    total = len(parts)
    for index, part in enumerate(parts, start=1):
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json",
            auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            data={
                "From": settings.TWILIO_WHATSAPP_FROM,
                "To": to,
                "Body": part,
            },
            timeout=settings.WHATSAPP_MEDIA_TIMEOUT_SECONDS,
        )
        if not response.ok:
            logger.error(
                "Twilio REST send failed part=%s/%s status=%s body=%s",
                index,
                total,
                response.status_code,
                response.text[:500],
            )
            response.raise_for_status()
        logger.info(
            "WhatsApp REST message sent to=%s part=%s/%s chars=%s",
            normalize_whatsapp_number(to),
            index,
            total,
            len(part),
        )


async def _send_whatsapp_message(to: str, body: str) -> None:
    await asyncio.to_thread(_send_whatsapp_message_sync, to, body)


def _send_whatsapp_single_message_sync(to: str, body: str) -> None:
    """Send exactly ONE WhatsApp message (prompt output must already fit Twilio limit)."""
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        raise RuntimeError("Twilio credentials not configured")

    message = (body or "").strip()
    response = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json",
        auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
        data={
            "From": settings.TWILIO_WHATSAPP_FROM,
            "To": to,
            "Body": message,
        },
        timeout=settings.WHATSAPP_MEDIA_TIMEOUT_SECONDS,
    )
    if not response.ok:
        logger.error(
            "Twilio single send failed chars=%s status=%s body=%s",
            len(message),
            response.status_code,
            response.text[:500],
        )
        response.raise_for_status()
    logger.info(
        "WhatsApp single message sent to=%s chars=%s",
        normalize_whatsapp_number(to),
        len(message),
    )


async def _send_whatsapp_single_message(to: str, body: str) -> None:
    await asyncio.to_thread(_send_whatsapp_single_message_sync, to, body)


def _get_or_create_user(from_value: str):
    phone_digits = normalize_whatsapp_number(from_value)
    if not phone_digits:
        return None
    return db_service.get_or_create_user(
        telegram_id=int(phone_digits),
        first_name="WhatsApp",
        last_name=None,
        username=f"wa_{phone_digits[-10:]}",
    )




async def _answer_user_question(db_user, user_text: str, *, log_prefix: str = "normal") -> str:
    """
    Unified answer path matching Telegram: normal chat and /q both use hybrid
    SQL+vector retrieval, with general chat only as the last fallback.
    """
    if not db_user:
        return "Error: Could not identify user."

    is_normal_chat = log_prefix == "normal"

    # ── Multi-item paragraph detection (runs only for normal chat) ────────────
    if is_normal_chat:
        multi = nlp_service_v2.classify_multi_item_expense(user_text)
        if multi.get("is_multi_item") and multi.get("items"):
            saved_items = []
            failed_items = []
            for item in multi["items"]:
                try:
                    # Build rich text per item so _extract_amount_from_text finds the amount
                    amt = item.get("amount")
                    vendor = item.get("vendor")
                    qty = item.get("quantity")
                    desc = item.get("description", "")
                    parts = []
                    if amt is not None:
                        parts.append(f"Rs {amt:.0f}")
                    parts.append(desc)
                    if vendor:
                        parts.append(f"from {vendor}")
                    if qty:
                        parts.append(f"({qty})")
                    item_text = " ".join(parts)

                    entry = db_service.save_user_text_entry(
                        user_id=db_user.id,
                        user_text=item_text,
                        intent_tag="expense_related_message",
                        expense_category=item.get("category") or "Other",
                        source="whatsapp",
                    )
                    saved_items.append({
                        **item,
                        "category": getattr(entry, "expense_category", None) or item.get("category") or "Other",
                        "amount": getattr(entry, "amount", None) if getattr(entry, "amount", None) is not None else item.get("amount"),
                    })
                except Exception:
                    logger.exception(
                        "Failed to save multi-item expense entry '%s' for user %s",
                        item.get("description", ""),
                        db_user.id,
                    )
                    failed_items.append(item)

            # AI-generated confirmation — emoji placement driven by prompt, not code
            answer = nlp_service_v2.generate_expense_save_confirmation(
                saved_items=saved_items,
                user_emotion=multi.get("user_emotion", "neutral"),
                user_id=db_user.id,
            )
            if failed_items:
                failed_names = ", ".join(it.get("description", "?") for it in failed_items)
                answer += f"\n\n⚠️ Ye items save nahi ho sake: {failed_names}"
            nlp_service_v2._add_to_history(db_user.id, user_text, answer)
            return answer
    # ─────────────────────────────────────────────────────────────────────────

    expense_decision = (
        nlp_service_v2.classify_plain_text_expense(user_text)
        if is_normal_chat
        else {"should_store": False, "category": "Other", "user_emotion": "neutral"}
    )
    if is_normal_chat and expense_decision.get("should_store"):
        try:
            entry = db_service.save_user_text_entry(
                user_id=db_user.id,
                user_text=user_text,
                intent_tag="expense_related_message",
                expense_category=expense_decision.get("category") or "Other",
                source="whatsapp",
            )
            category = getattr(entry, "expense_category", None) or expense_decision.get("category") or "Other"
            amount = getattr(entry, "amount", None)
            # AI-generated confirmation — emoji placement driven by prompt, not code
            answer = nlp_service_v2.generate_expense_save_confirmation(
                saved_items=[{
                    "description": user_text[:120],
                    "category": category,
                    "amount": amount,
                    "vendor": None,
                }],
                user_emotion=expense_decision.get("user_emotion", "neutral"),
                user_id=db_user.id,
            )
            nlp_service_v2._add_to_history(db_user.id, user_text, answer)
            return answer
        except Exception:
            logger.exception("Failed to persist WhatsApp expense text for user %s", db_user.id)
            return "Expense entry save nahi ho paayi. Please thodi der baad try karein."

    try:
        result = nlp_service_v2.ask_ai(user_text, db_user.id)
        answer = (result.get("ai_response") or "").strip()
        if answer:
            logger.info(
                "WhatsApp %s answered via hybrid pipeline (hybrid=%s, fallback=%s)",
                log_prefix,
                result.get("hybrid"),
                result.get("fallback"),
            )
            return answer
    except Exception:
        logger.exception("WhatsApp %s: NLP v2 pipeline failed", log_prefix)

    history = nlp_service_v2._get_conversation_context(db_user.id)
    prompt = f"{history}\n\nUser: {user_text}" if history else user_text
    emotion = expense_decision.get("user_emotion") or "neutral"
    system_prompt = telegram_settings.SYSTEM_PROMPT + (
        f"\n\nReply tone hint: {emotion}. User data is private; never reveal other users' data."
    )
    answer = await openai_service.ask(
        user_prompt=prompt,
        system_prompt=system_prompt,
        chat_id=None,
        use_memory=False,
    )
    nlp_service_v2._add_to_history(db_user.id, user_text, answer)
    return answer


def _help_text() -> str:
    return (
        "WhatsApp bot is live.\n\n"
        "Commands:\n"
        "/help - Show commands\n"
        "/summary - Document summary\n"
        "/mydocs - Recent documents\n"
        "/q <question> - Ask about your documents\n"
        "/clear - Clear chat memory\n\n"
        "You can also send text expenses, images, or PDFs. For uploads, reply CONFIRM <id> to save."
    )


def _summary_text(db_user) -> str:
    stats = db_service.get_user_summary_stats(db_user.id)
    docs = db_service.get_user_documents(db_user.id, limit=5)
    if not docs:
        return "No documents yet. Send an image or PDF receipt to extract and save it."
    lines = [
        "Your Document Summary",
        f"Docs: {stats.get('total_documents', 0)}",
        f"Total Spend: Rs {stats.get('total_amount', 0):,.2f}",
        f"Unique Vendors: {stats.get('unique_vendors', 0)}",
        "",
        "Recent Documents:",
    ]
    for doc in docs:
        amount = f"Rs {doc.total_amount:,.2f}" if doc.total_amount else "N/A"
        lines.append(f"- #{doc.id} {doc.title or doc.file_name or 'Untitled'} | {amount}")
    return "\n".join(lines)


def _mydocs_text(db_user) -> str:
    docs = db_service.get_user_documents(db_user.id, limit=20)
    if not docs:
        return "No documents yet. Send an image or PDF receipt to extract and save it."
    lines = ["Recent Documents:"]
    for doc in docs:
        amount = f"Rs {doc.total_amount:,.2f}" if doc.total_amount else "N/A"
        lines.append(f"- #{doc.id} {doc.title or doc.file_name or 'Untitled'} | {amount}")
    return "\n".join(lines)


def _twilio_account_sid_from_media_url(media_url: str) -> str | None:
    match = re.search(r"/Accounts/(AC[a-f0-9]{32})/", media_url or "")
    return match.group(1) if match else None


def _download_twilio_media(media_url: str) -> bytes:
    if not media_url:
        raise ValueError("Twilio media URL is missing from webhook payload")
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        raise RuntimeError("Twilio credentials are required to download WhatsApp media")

    media_account_sid = _twilio_account_sid_from_media_url(media_url)
    if media_account_sid and media_account_sid != settings.TWILIO_ACCOUNT_SID:
        raise RuntimeError(
            "Twilio credential mismatch: webhook media belongs to account "
            f"{media_account_sid}, but TWILIO_ACCOUNT_SID is {settings.TWILIO_ACCOUNT_SID}. "
            "Update TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env to match the active Twilio account."
        )

    response = requests.get(
        media_url,
        auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
        timeout=settings.WHATSAPP_MEDIA_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "Twilio rejected media download (401 Unauthorized). "
            "Verify TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env match the account "
            f"that owns WhatsApp number {settings.TWILIO_WHATSAPP_FROM}."
        )
    response.raise_for_status()
    return response.content


async def _build_media_review_reply(
    *,
    db_user,
    file_bytes: bytes,
    mime_type: str,
    caption: str,
    ocr_raw: str,
) -> str:
    """Build pending-document review card with full OCR text for user confirm/edit."""
    file_size = len(file_bytes)
    file_name = "whatsapp-upload.pdf" if mime_type == "application/pdf" else "whatsapp-upload.jpg"
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    dhash = None
    phash = None

    if mime_type.startswith("image/"):
        dhash, phash = generate_image_hashes(file_bytes)
        duplicate = db_service.find_duplicate_image_for_user(
            db_user.id,
            content_sha256=content_sha256,
            dhash=dhash,
            phash=phash,
        )
        if duplicate:
            return "Duplicate image detected. This upload was not saved again."

    if mime_type.startswith("image/"):
        duplicate_after_ocr = db_service.find_duplicate_by_extracted_fingerprint(db_user.id, ocr_raw)
        if duplicate_after_ocr:
            return "Duplicate document detected after OCR. This upload was not saved again."

    confidence = _get_confidence(ocr_raw)
    pending = db_service.create_pending_document(
        user_id=db_user.id,
        file_name=file_name,
        mime_type=mime_type,
        file_size=file_size,
        extracted_json=ocr_raw,
        confidence_overall=confidence,
        user_input_text=caption or None,
        source="whatsapp",
        content_sha256=content_sha256,
        dhash=dhash,
        phash=phash,
        status="pending",
    )
    return await build_whatsapp_review_message(ocr_raw, pending.id)


async def _handle_media_upload(form: dict[str, str], db_user) -> str:
    mime_type = form.get("MediaContentType0", "")
    if mime_type not in SUPPORTED_MEDIA_TYPES:
        return f"I can process JPG, PNG, WEBP, and PDF files. Received: {mime_type or 'unknown'}"

    file_bytes = _download_twilio_media(form.get("MediaUrl0", ""))
    caption = (form.get("Body") or "").strip()

    result = await ocr_service.extract_data(
        image_bytes=file_bytes,
        mime_type=mime_type,
        user_input_text=caption or None,
    )

    try:
        result_data = json.loads(result)
    except Exception:
        result_data = {}
    if result_data.get("status") == "unreadable":
        return result_data.get(
            "message",
            "I couldn't clearly understand the uploaded image. Please re-upload a clearer image.",
        )

    return await _build_media_review_reply(
        db_user=db_user,
        file_bytes=file_bytes,
        mime_type=mime_type,
        caption=caption,
        ocr_raw=result,
    )


async def _redis():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    _REDIS_CLIENT = await get_redis_client(settings.REDIS_URL)
    return _REDIS_CLIENT


async def _redis_optional():
    try:
        return await _redis()
    except Exception as exc:
        logger.warning("Redis unavailable (%s); continuing with in-memory dedupe only", exc)
        return None


def _local_claim_send_once(key: str, ttl_seconds: int) -> bool:
    now = time.time()
    stale = [k for k, ts in _LOCAL_SEND_CLAIMS.items() if now - ts > ttl_seconds]
    for k in stale:
        _LOCAL_SEND_CLAIMS.pop(k, None)
    if key in _LOCAL_SEND_CLAIMS:
        return False
    _LOCAL_SEND_CLAIMS[key] = now
    return True


async def _claim_process_once(client, ref: WhatsappRequestRef) -> bool:
    ttl = settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS
    if client is not None:
        return await claim_process_once(client, ref, ttl_seconds=ttl)
    return _local_claim_send_once(f"{ref.key}:processing", ttl)


async def _claim_send_once(client, ref: WhatsappRequestRef) -> bool:
    ttl = settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS
    if client is not None:
        return await claim_send_once(client, ref, ttl_seconds=ttl)
    return _local_claim_send_once(ref.sent_key, ttl)


async def _process_media_request_and_send(
    *,
    from_value: str,
    from_number: str,
    request_id: str,
    message_sid: str,
    form: dict[str, str],
    db_user,
) -> None:
    ref = WhatsappRequestRef(phone_number=from_number, request_id=request_id)
    client = None
    try:
        client = await _redis_optional()

        if not await _claim_process_once(client, ref):
            logger.info("Skipping duplicate processing for %s", ref.key)
            return

        if client is not None:
            await init_request_state(
                client,
                ref,
                ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS,
                from_address=from_value,
                message_sid=message_sid,
            )

        mime_type = form.get("MediaContentType0", "")
        if mime_type not in SUPPORTED_MEDIA_TYPES:
            msg = f"I can process JPG, PNG, WEBP, and PDF files. Received: {mime_type or 'unknown'}"
            if client is not None:
                await mark_error(client, ref, msg, ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS)
            if await _claim_send_once(client, ref):
                await _send_whatsapp_single_message(from_value, msg)
            return

        file_bytes = _download_twilio_media(form.get("MediaUrl0", ""))
        caption = (form.get("Body") or "").strip()

        # Phase 2: OCR — no WhatsApp send until complete.
        ocr_raw = await ocr_service.extract_data(
            image_bytes=file_bytes,
            mime_type=mime_type,
            user_input_text=caption or None,
        )
        if client is not None:
            await patch_state(
                client,
                ref,
                {"status": "ocr_complete", "ocr": ocr_raw},
                ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS,
            )

        try:
            ocr_data = json.loads(ocr_raw)
        except Exception:
            ocr_data = {}
        if isinstance(ocr_data, dict) and ocr_data.get("status") == "unreadable":
            msg = ocr_data.get(
                "message",
                "I couldn't clearly understand the uploaded image. Please re-upload a clearer image.",
            )
            if client is not None:
                await mark_error(client, ref, msg, ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS)
            if await _claim_send_once(client, ref):
                await _send_whatsapp_single_message(from_value, msg)
            return

        # Phase 3–5: build full review card in Redis — send only when ready.
        final_message = await _build_media_review_reply(
            db_user=db_user,
            file_bytes=file_bytes,
            mime_type=mime_type,
            caption=caption,
            ocr_raw=ocr_raw,
        )

        extracted_fields = extract_basic_fields_from_ocr_json(ocr_raw)
        if client is not None:
            await patch_state(
                client,
                ref,
                {
                    "status": "card_ready",
                    "extraction": extracted_fields,
                    "categorization": {"category": extracted_fields.get("category")},
                    "final_message": final_message,
                },
                ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS,
            )
            await mark_completed(
                client,
                ref,
                final_message=final_message,
                ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS,
            )

        # Single Twilio send after all extraction + card aggregation is done.
        if not await _claim_send_once(client, ref):
            logger.info("Skipping duplicate send for %s", ref.key)
            return
        await _send_whatsapp_single_message(from_value, final_message)
    except Exception:
        logger.exception("WhatsApp media pipeline failed request=%s", ref.key)
        msg = "Sorry, I could not process that upload right now. Please try again."
        if client is not None:
            try:
                await mark_error(client, ref, msg, ttl_seconds=settings.WHATSAPP_REQUEST_STATE_TTL_SECONDS)
            except Exception:
                pass
        if await _claim_send_once(client, ref):
            try:
                await _send_whatsapp_single_message(from_value, msg)
            except Exception:
                pass
    finally:
        if client is not None:
            try:
                await cleanup_request_state(client, ref)
            except Exception:
                pass


def _get_confidence(extracted_json: str) -> float:
    try:
        data = json.loads(extracted_json)
        confidence = data.get("confidence")
        if isinstance(confidence, dict):
            return float(confidence.get("overall") or 0.0)
        if confidence is not None:
            return float(confidence)
    except Exception:
        pass
    return 0.0


async def _confirm_pending(pending_id: int, db_user) -> str:
    pending = db_service.get_pending_document_by_id(pending_id, db_user.id)
    if not pending:
        return "Pending document not found or already handled."
    doc = db_service.confirm_pending_document(pending_id)
    if not doc:
        return "Failed to save document. Please try again."
    indexed = DatabaseService.index_document_in_vector_store(doc)
    if not indexed:
        indexed = DatabaseService.reindex_document_vector(doc.id, doc.user_id)
    note = " Indexed for search." if indexed else " Saved, but vector indexing failed."
    vendor = doc.vendor_name or doc.title or "Unknown vendor"
    amount = f"₹{doc.total_amount:,.2f}" if doc.total_amount else "N/A"
    history_note = f"Saved receipt from {vendor}, amount {amount}, document ID {doc.id}."
    nlp_service_v2._add_to_history(doc.user_id, f"Confirmed WhatsApp upload {pending_id}.", history_note)
    return f"Document saved. ID: {doc.id}.{note}"


async def _edit_pending(pending_id: int, corrected_json: str, db_user) -> str:
    pending = db_service.get_pending_document_by_id(pending_id, db_user.id)
    if not pending:
        return "Pending document not found or already handled."
    try:
        json.loads(corrected_json)
    except json.JSONDecodeError:
        return "Invalid JSON. Please send: EDIT <id> {corrected_json}"
    updated = db_service.update_pending_document(pending_id, corrected_json)
    if not updated:
        return "Failed to update pending document."
    return await _confirm_pending(pending_id, db_user)


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request) -> Response:
    form = await _twilio_form(request)
    message_sid = form.get("MessageSid", "")
    if not form.get("From") or ("Body" not in form and not form.get("NumMedia")):
        logger.info("Ignoring non-inbound WhatsApp callback sid=%s keys=%s", message_sid or "-", sorted(form.keys()))
        return _xml_response()
    cached = _cached_response_for(message_sid)
    if cached:
        logger.info("Returning cached WhatsApp response for MessageSid=%s", message_sid)
        return cached

    if not _is_valid_twilio_signature(request, form):
        logger.warning("Rejected WhatsApp webhook with invalid Twilio signature")
        return _xml_response("Invalid webhook signature.")

    from_value = form.get("From", "")
    from_number = normalize_whatsapp_number(from_value)

    if not settings.is_number_allowed(from_value):
        logger.warning("Unauthorized WhatsApp message from %s", from_value)
        return _xml_response("You are not authorized to use this bot.")

    db_user = _get_or_create_user(from_value)
    if not db_user:
        return _xml_response("Could not identify WhatsApp user.")

    body = (form.get("Body") or "").strip()
    logger.info("Incoming WhatsApp message from %s: %s", from_number, body[:200])

    try:
        if len(body) > settings.WHATSAPP_MAX_BODY_LENGTH:
            return _cacheable_xml_response(
                message_sid,
                f"Message too long. Please keep it under {settings.WHATSAPP_MAX_BODY_LENGTH} characters.",
            )

        media_count = int(form.get("NumMedia") or "0")
        if media_count > settings.WHATSAPP_MAX_MEDIA_COUNT:
            return _cacheable_xml_response(
                message_sid,
                f"Please send one file at a time. Received {media_count} files.",
            )

        if media_count > 0:
            # Avoid Twilio webhook timeout: process asynchronously and send ONE final aggregated message via REST.
            request_id = message_sid or uuid.uuid4().hex
            _cacheable_xml_response(message_sid)  # populate cache to prevent duplicate retries scheduling
            asyncio.create_task(
                _process_media_request_and_send(
                    from_value=from_value,
                    from_number=from_number,
                    request_id=request_id,
                    message_sid=message_sid,
                    form=form,
                    db_user=db_user,
                )
            )
            return _cacheable_xml_response(message_sid)

        async def _background_answer_and_send(user_text: str, *, log_prefix: str) -> None:
            try:
                answer = await _answer_user_question(db_user, user_text, log_prefix=log_prefix)
                await _send_whatsapp_message(from_value, answer)
                logger.info("WhatsApp async answer sent sid=%s prefix=%s", message_sid or "-", log_prefix)
            except Exception:
                logger.exception("WhatsApp async answer failed sid=%s prefix=%s", message_sid or "-", log_prefix)

        confirm_match = re.match(r"^(?:confirm|save)\s+(\d+)\s*$", body, flags=re.IGNORECASE)
        if confirm_match:
            return _cacheable_xml_response(message_sid, await _confirm_pending(int(confirm_match.group(1)), db_user))

        edit_match = re.match(r"^edit\s+(\d+)\s+(.+)$", body, flags=re.IGNORECASE | re.DOTALL)
        if edit_match:
            return _cacheable_xml_response(
                message_sid,
                await _edit_pending(int(edit_match.group(1)), edit_match.group(2).strip(), db_user)
            )

        lower = body.lower()
        if lower in {"/start", "start", "hi", "hello"}:
            return _cacheable_xml_response(message_sid, _help_text())
        if lower in {"/help", "help"}:
            return _cacheable_xml_response(message_sid, _help_text())
        if lower == "/clear":
            openai_service.clear_conversation(f"whatsapp:{db_user.id}")
            nlp_service_v2.clear_user_history(db_user.id)
            return _cacheable_xml_response(message_sid, "Conversation memory cleared.")
        if lower == "/summary":
            return _cacheable_xml_response(message_sid, _summary_text(db_user))
        if lower == "/mydocs":
            return _cacheable_xml_response(message_sid, _mydocs_text(db_user))
        if lower.startswith("/q"):
            query = body[2:].strip()
            if not query:
                return _cacheable_xml_response(message_sid, "Please send /q followed by your question.")
            # Avoid Twilio webhook timeout by replying via REST asynchronously.
            _cacheable_xml_response(message_sid)  # populate cache to prevent duplicate retries scheduling
            asyncio.create_task(_background_answer_and_send(query, log_prefix="/q"))
            return _cacheable_xml_response(message_sid)
        if not body:
            return _cacheable_xml_response(message_sid, _help_text())

        # Normal chat: avoid webhook timeouts (especially under 429 retries) by sending answer via REST asynchronously.
        _cacheable_xml_response(message_sid)  # populate cache to prevent duplicate retries scheduling
        asyncio.create_task(_background_answer_and_send(body, log_prefix="normal"))
        return _cacheable_xml_response(message_sid)
    except Exception:
        logger.exception("WhatsApp webhook failed for %s", from_number)
        return _cacheable_xml_response(message_sid, "Sorry, I could not process that just now. Please try again.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
