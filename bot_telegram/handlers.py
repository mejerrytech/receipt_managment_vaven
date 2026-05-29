import io
import json
import logging
import time
import hashlib
import os
from typing import Any, Dict, List
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes
from shared.openai_client import OpenAIService
from shared.database import DatabaseService
from shared.nlp_sql_service_v2 import get_nlp_sql_service_v2
from shared.rag_service import get_rag_service
from shared.ocr_service import get_ocr_service
from shared.image_hash_service import generate_image_hashes
from bot_telegram.config import settings

try:
    from shared.tasks.ocr_tasks import process_pending_ocr
    OCR_QUEUE_AVAILABLE = True
except Exception:
    process_pending_ocr = None
    OCR_QUEUE_AVAILABLE = False

logger = logging.getLogger("telegram_handlers")
openai_service = OpenAIService()
db_service = DatabaseService()
ocr_service = get_ocr_service()

# New orchestrated services with GPT-4o primary + Anthropic fallback
nlp_service_v2 = get_nlp_sql_service_v2()
rag_service = get_rag_service()
OCR_MAX_INFLIGHT_PER_USER = int(os.getenv("OCR_MAX_INFLIGHT_PER_USER", "3"))
if not OCR_QUEUE_AVAILABLE:
    logger.warning("Celery OCR queue is unavailable; using synchronous OCR fallback.")

def _get_confidence(extracted_json: str) -> float:
    """Extract confidence score from OCR JSON result."""
    try:
        data = json.loads(extracted_json)
        # Check common confidence locations
        if "confidence" in data:
            if isinstance(data["confidence"], dict):
                return data["confidence"].get("overall", 0.0)
            return float(data["confidence"])
        # Check nested structure
        if "confidence" in data and isinstance(data.get("confidence"), dict):
            return data["confidence"].get("overall", 0.0)
        return 1.0  # Default to high confidence if not specified
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0  # Low confidence if we can't parse


def _safe_json_loads(extracted_json: str) -> dict:
    """Parse OCR JSON safely and return dict."""
    try:
        parsed = json.loads(extracted_json)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _compact_ocr_payload_for_summary(extracted_json: str) -> Dict[str, Any]:
    """Prepare compact OCR payload for dynamic UI summarization."""
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


def _remember_saved_document_for_qa(user_id: int, extracted_json: str, label: str = "saved document") -> None:
    """Add the saved OCR payload to NLP context so immediate follow-ups can resolve it."""
    try:
        context_payload = _compact_ocr_payload_for_summary(extracted_json)
        if not context_payload.get("fields") and not context_payload.get("items"):
            parsed = _safe_json_loads(extracted_json)
            context_payload = {"extracted_data": parsed} if parsed else {}
        if not context_payload:
            return

        context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
        nlp_service_v2._add_to_history(
            user_id,
            f"Uploaded and saved {label}.",
            "Saved document context for future questions:\n" + context_json,
        )
    except Exception:
        logger.exception("Failed to add saved document to QA context for user %s", user_id)


async def _build_upload_preview_card(extracted_json: str, confidence: float, doc_label: str, file_name: str = "") -> str:
    """
    Build dynamic OCR summary card through GPT.
    Only shows available data; hides missing keys automatically.
    """
    is_high_conf = confidence > 0.8

    compact_payload = _compact_ocr_payload_for_summary(extracted_json)
    payload_json = json.dumps(compact_payload, ensure_ascii=False, indent=2)

    system_prompt = """You create concise OCR summary cards for Telegram in Markdown.

Rules:
1. Use only provided data. Never invent missing values.
2. Skip missing fields entirely (do not print N/A).
3. Keep response compact and readable.
4. If items exist, include bullet points with qty/unit/amount only when present.
5. Do not include these labels in output: "Image OCR Summary Card", "Confidence", "File", "Unordered List".
6. Do not include JSON, code blocks, or technical metadata."""

    user_prompt = f"""Generate only the card body.

Data:
{payload_json}"""

    try:
        llm_summary = await openai_service.ask(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            use_memory=False
        )
    except Exception:
        llm_summary = ""

    llm_summary = (llm_summary or "").strip()
    if not llm_summary:
        # Deterministic fallback with dynamic fields only.
        fields = compact_payload.get("fields", {})
        items = compact_payload.get("items", [])
        lines: List[str] = [f"**{k.replace('_', ' ').title()}:** {v}" for k, v in fields.items()]
        if items:
            lines.append("")
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

    # Telegram doesn't support true border colors; use green/orange themed border lines.
    border = "🟢────────────────────────" if is_high_conf else "🟠────────────────────────"
    return f"{border}\n{llm_summary}\n{border}"


def _get_or_create_user(update: Update):
    """Get or create user from Telegram update. Returns user object from DB."""
    if not update.effective_user:
        return None
    
    user = update.effective_user
    telegram_id = user.id
    first_name = user.first_name
    last_name = user.last_name
    username = user.username
    
    return db_service.get_or_create_user(
        telegram_id=telegram_id,
        first_name=first_name,
        last_name=last_name,
        username=username
    )


def _is_user_allowed(update: Update) -> bool:
    """Check if user is in allowlist."""
    user_id = str(update.effective_user.id) if update.effective_user else None
    if not user_id:
        return False
    return settings.is_user_allowed(user_id)


async def _reply_in_chunks(update: Update, text: str, max_length: int = 3500) -> None:
    """Split long messages into chunks and send separately."""
    if len(text) <= max_length:
        await update.message.reply_text(text)
        return
    
    # Split by paragraphs first, then by sentences if needed
    chunks = []
    current_chunk = ""
    
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_length:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If single paragraph is too long, split by sentences
            if len(paragraph) > max_length:
                sentences = paragraph.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_length:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
            else:
                current_chunk = paragraph + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send chunks
    for i, chunk in enumerate(chunks):
        if chunk:
            prefix = f"(Part {i+1}/{len(chunks)})\n\n" if len(chunks) > 1 else ""
            await update.message.reply_text(prefix + chunk)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    start_time = time.time()
    
    # Check user allowlist
    if not _is_user_allowed(update):
        logger.warning(f"Unauthorized access attempt from user: {update.effective_user.id}")
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    latency = time.time() - start_time
    logger.info(f"User {update.effective_user.id} started bot (latency: {latency:.3f}s)")
    
    await update.message.reply_text(
        " **Telegram Bot is Live!**\n\n"
        "I can help you with:\n"
        "• General questions\n"
        "• Conversation memory\n\n"
        "Use /help for available commands.",
        parse_mode="Markdown"
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Check user allowlist
    if not _is_user_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    help_text = (
        " **Available Commands**\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/summary - Get summary of your documents (total docs, amount, vendors)\n"
        "/q <question> - Ask about your documents (e.g., 'show my invoices')\n"
        "/mydocs - View your uploaded documents\n"
        "/clear - Clear conversation memory\n\n"
        "**Features**\n"
        "• Upload **photos** or **images** for OCR data extraction\n"
        "• Upload **PDF** documents for text/data extraction\n"
        "• Get document summary with /summary\n"
        "• Query documents with natural language using /q\n"
        "• All documents are saved with extracted data\n"
        "• Conversation memory enabled\n"
        "• 45s timeout with 2 retries\n"
        "• Auto-chunking for long responses\n\n"
        "Just send any message to chat with AI, or upload an image/PDF to extract data!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear conversation history for the chat."""
    if not _is_user_allowed(update):
        await update.message.reply_text(" You are not authorized to use this bot.")
        return
    
    chat_id = str(update.effective_chat.id)
    openai_service.clear_conversation(chat_id)
    db_user = _get_or_create_user(update)
    if db_user:
        try:
            nlp_service_v2.clear_user_history(db_user.id)
        except Exception:
            logger.exception("Failed clearing NLP SQL history for user %s", db_user.id)
    await update.message.reply_text("Conversation memory cleared (OpenAI + /q context).")


async def websearch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /websearch command (disabled unless WEB_SEARCH_ENABLED=true)."""
    if not settings.WEB_SEARCH_ENABLED:
        await update.message.reply_text(
            "Web search is disabled. Ask about your uploaded receipts/invoices in normal chat or use /q — "
            "answers come from your SQL + vector data only."
        )
        return

    start_time = time.time()
    
    # Check user allowlist
    if not _is_user_allowed(update):
        logger.warning(f"Unauthorized websearch attempt from user: {update.effective_user.id}")
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    if not update.message:
        return
    
    # Get the search query from command args
    query = " ".join(context.args) if context.args else None
    
    if not query:
        await update.message.reply_text(
            "**Web Search**\n\n"
            "Please provide a search query.\n"
            "Example: `/websearch latest AI news`",
            parse_mode="Markdown"
        )
        return
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    try:
        # Use web search with retries and timeout
        answer = await openai_service.web_search(
            user_prompt=query,
            system_prompt="You are a helpful assistant with access to current web information. Provide accurate, up-to-date answers based on web search results."
        )
        
        # Send response (chunked if too long)
        header = f"🔍 **Web Search: {query[:50]}{'...' if len(query) > 50 else ''}**\n\n"
        full_response = header + answer
        
        await _reply_in_chunks(update, full_response, settings.MAX_MESSAGE_LENGTH)
        
        latency = time.time() - start_time
        logger.info(f"Web search completed for user {update.effective_user.id} in {latency:.2f}s")
        
    except Exception:
        logger.exception("Web search failed for user %s", update.effective_user.id)
        await update.message.reply_text(
            "Web search is temporarily unavailable. Please try again in a moment."
        )


async def _answer_user_question(
    update: Update,
    db_user,
    user_text: str,
    *,
    log_prefix: str = "query",
) -> str:
    """
    Unified answer path for normal chat and /q: hybrid SQL+vector via NLP v2,
    shared per-user conversation memory (user_id scoped).
    """
    if not db_user:
        return "Error: Could not identify user."

    is_normal_chat = log_prefix == "normal"
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
            )
            category = getattr(entry, "expense_category", None) or expense_decision.get("category") or "Other"
            amount = getattr(entry, "amount", None)
            amount_part = f" Amount: ₹{amount:,.2f}." if amount is not None else ""
            answer = f"Saved as {category} expense.{amount_part}"
            nlp_service_v2._add_to_history(db_user.id, user_text, answer)
            return answer
        except Exception:
            logger.exception("Failed to persist expense text for user %s", db_user.id)
            return "Expense entry save nahi ho paayi. Please thodi der baad try karein."

    try:
        result = nlp_service_v2.ask_ai(user_text, db_user.id)
        answer = (result.get("ai_response") or "").strip()
        if answer:
            logger.info(
                "%s answered via hybrid pipeline (hybrid=%s, fallback=%s)",
                log_prefix,
                result.get("hybrid"),
                result.get("fallback"),
            )
            return answer
    except Exception:
        logger.exception("%s: NLP v2 pipeline failed", log_prefix)

    # Last resort: general chat with the same NLP conversation history injected.
    hist = nlp_service_v2._get_conversation_context(db_user.id)
    prompt = user_text
    if hist:
        prompt = f"{hist}\n\nUser: {user_text}"
    emotion = expense_decision.get("user_emotion") or "neutral"
    system_prompt = settings.SYSTEM_PROMPT + (
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


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular messages."""
    start_time = time.time()

    # Check user allowlist
    if not _is_user_allowed(update):
        logger.warning(f"Unauthorized message from user: {update.effective_user.id}")
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Get or create user in database
    db_user = _get_or_create_user(update)
    if db_user:
        logger.info(f"User {db_user.telegram_id} (@{db_user.username}) interacting")

    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    logger.info("Incoming user message for intent check: '%s'", user_text)
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    try:
        answer = await _answer_user_question(update, db_user, user_text, log_prefix="normal")
        await _reply_in_chunks(update, answer, settings.MAX_MESSAGE_LENGTH)
        
        latency = time.time() - start_time
        logger.info(
            f"Message processed for user {update.effective_user.id} in {latency:.2f}s"
        )
        
    except Exception:
        logger.exception("Message processing failed for user %s", update.effective_user.id)
        await update.message.reply_text(
            "Sorry — I could not process that message just now. Please try again."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler."""
    logger.error(f"Update {update} caused error: {context.error}", exc_info=True)

    if update and update.message:
        try:
            await update.message.reply_text(
                " An unexpected error occurred. Please try again later."
            )
        except Exception:
            pass  # Ignore errors in error handler


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo uploads with OCR data extraction."""
    start_time = time.time()

    # Check user allowlist
    if not _is_user_allowed(update):
        logger.warning(f"Unauthorized photo upload from user: {update.effective_user.id}")
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Get or create user in database
    db_user = _get_or_create_user(update)
    if not db_user:
        await update.message.reply_text("Error: Could not identify user.")
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        # Get the largest photo (best quality)
        photo = update.message.photo[-1]
        user_input_text = (update.message.caption or "").strip() if update.message else ""
        telegram_file_id = photo.file_id
        telegram_file_unique_id = photo.file_unique_id
        file = await photo.get_file()

        # Download image bytes
        image_bytes = await file.download_as_bytearray()
        file_size = len(image_bytes)
        content_sha256 = hashlib.sha256(bytes(image_bytes)).hexdigest()
        dhash, phash = generate_image_hashes(bytes(image_bytes))

        duplicate = db_service.find_duplicate_image_for_user(
            db_user.id,
            telegram_file_unique_id=telegram_file_unique_id,
            content_sha256=content_sha256,
            dhash=dhash,
            phash=phash
        )
        if duplicate:
            await update.message.reply_text(
                "⚠️ Duplicate image detected.\n"
                "Ye image aap pehle hi upload kar chuke ho. Naya save nahi kiya gaya."
            )
            return

        logger.info(f"Processing photo from user {db_user.telegram_id} (@{db_user.username}), size: {file_size} bytes")

        inflight = db_service.count_user_inflight_pending_documents(db_user.id)
        if inflight >= OCR_MAX_INFLIGHT_PER_USER:
            await update.message.reply_text(
                f"⚠️ Aapke {inflight} OCR jobs already processing me hain.\n"
                "Thoda wait karke fir upload karein."
            )
            return

        if OCR_QUEUE_AVAILABLE and process_pending_ocr is not None:
            # Store processing document in database before queueing
            pending = db_service.create_pending_document(
                user_id=db_user.id,
                file_name="photo.jpg",
                mime_type="image/jpeg",
                file_size=file_size,
                extracted_json="{}",
                confidence_overall=None,
                user_input_text=user_input_text or None,
                source='telegram',
                telegram_chat_id=update.effective_chat.id,
                telegram_file_id=telegram_file_id,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                status='processing'
            )
            job = process_pending_ocr.delay(pending.id, db_user.id)
            db_service.set_pending_job_id(pending.id, job.id)

            ack = await update.message.reply_text(
                "Upload received. OCR processing started in background.\n"
                "Result aate hi confirm/edit buttons ke saath message aa jayega."
            )
            db_service.set_pending_telegram_message_id(pending.id, ack.message_id)
        else:
            # Fallback to synchronous OCR when Celery is unavailable.
            result = await ocr_service.extract_data(
                image_bytes=bytes(image_bytes),
                mime_type="image/jpeg",
                user_input_text=user_input_text or None
            )
            duplicate_after_ocr = db_service.find_duplicate_by_extracted_fingerprint(db_user.id, result)
            if duplicate_after_ocr:
                await update.message.reply_text(
                    "⚠️ Duplicate image detected.\n"
                    "Ye document pehle se hai (content match mila), naya save nahi kiya gaya."
                )
                return

            confidence = _get_confidence(result)
            pending = db_service.create_pending_document(
                user_id=db_user.id,
                file_name="photo.jpg",
                mime_type="image/jpeg",
                file_size=file_size,
                extracted_json=result,
                confidence_overall=confidence,
                user_input_text=user_input_text or None,
                source='telegram',
                telegram_chat_id=update.effective_chat.id,
                telegram_file_id=telegram_file_id,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                status='pending'
            )
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Confirm & Save", callback_data=f"confirm:{pending.id}")],
                [InlineKeyboardButton("✏️ Edit Data", callback_data=f"edit:{pending.id}")]
            ])
            preview_card = await _build_upload_preview_card(
                extracted_json=result,
                confidence=confidence,
                doc_label="Image",
                file_name="photo.jpg"
            )
            await update.message.reply_text(preview_card, reply_markup=keyboard, parse_mode="Markdown")

        latency = time.time() - start_time
        logger.info(f"Photo processed for user {db_user.telegram_id} in {latency:.2f}s, awaiting confirmation")

    except Exception:
        logger.exception("Photo processing failed for user %s", update.effective_user.id)
        await update.message.reply_text(
            "Sorry — I could not process this photo. Please try again with a clearer image."
        )


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads (PDFs and images) with OCR data extraction."""
    start_time = time.time()

    # Check user allowlist
    if not _is_user_allowed(update):
        logger.warning(f"Unauthorized document upload from user: {update.effective_user.id}")
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Get or create user in database
    db_user = _get_or_create_user(update)
    if not db_user:
        await update.message.reply_text("Error: Could not identify user.")
        return

    document = update.message.document
    mime_type = document.mime_type or ""
    file_name = document.file_name or "document"
    telegram_file_id = document.file_id
    telegram_file_unique_id = document.file_unique_id

    # Only process images and PDFs
    allowed_types = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
    if mime_type not in allowed_types:
        await update.message.reply_text(
            "📄 I can only process **images** (JPG, PNG, WEBP) and **PDF** files.\n"
            f"Received: `{mime_type}`"
        )
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        # Get file
        file = await document.get_file()
        user_input_text = (update.message.caption or "").strip() if update.message else ""
        file_bytes = await file.download_as_bytearray()
        file_size = len(file_bytes)
        content_sha256 = hashlib.sha256(bytes(file_bytes)).hexdigest()
        dhash = None
        phash = None

        if mime_type.startswith("image/"):
            dhash, phash = generate_image_hashes(bytes(file_bytes))
            duplicate = db_service.find_duplicate_image_for_user(
                db_user.id,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash
            )
            if duplicate:
                await update.message.reply_text(
                    "⚠️ Duplicate image detected.\n"
                    "Ye image aap pehle hi upload kar chuke ho. Naya save nahi kiya gaya."
                )
                return

        logger.info(f"Processing document from user {db_user.telegram_id} (@{db_user.username}): {file_name}, size: {file_size} bytes")

        # For PDFs, note that GPT-4 Vision works best with image-based PDFs
        display_type = "PDF" if mime_type == "application/pdf" else "Image"

        inflight = db_service.count_user_inflight_pending_documents(db_user.id)
        if inflight >= OCR_MAX_INFLIGHT_PER_USER:
            await update.message.reply_text(
                f"⚠️ Aapke {inflight} OCR jobs already processing me hain.\n"
                "Thoda wait karke fir upload karein."
            )
            return

        if OCR_QUEUE_AVAILABLE and process_pending_ocr is not None:
            # Store pending document in database
            pending = db_service.create_pending_document(
                user_id=db_user.id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                extracted_json="{}",
                confidence_overall=None,
                user_input_text=user_input_text or None,
                source='telegram',
                telegram_chat_id=update.effective_chat.id,
                telegram_file_id=telegram_file_id,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                status='processing'
            )
            job = process_pending_ocr.delay(pending.id, db_user.id)
            db_service.set_pending_job_id(pending.id, job.id)

            ack = await update.message.reply_text(
                f"{display_type} upload received. OCR processing started in background.\n"
                "Result aate hi confirm/edit buttons ke saath message aa jayega."
            )
            db_service.set_pending_telegram_message_id(pending.id, ack.message_id)
        else:
            result = await ocr_service.extract_data(
                image_bytes=bytes(file_bytes),
                mime_type=mime_type,
                user_input_text=user_input_text or None
            )
            if mime_type.startswith("image/"):
                duplicate_after_ocr = db_service.find_duplicate_by_extracted_fingerprint(db_user.id, result)
                if duplicate_after_ocr:
                    await update.message.reply_text(
                        "⚠️ Duplicate image detected."
                    )
                    return

            confidence = _get_confidence(result)
            pending = db_service.create_pending_document(
                user_id=db_user.id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                extracted_json=result,
                confidence_overall=confidence,
                user_input_text=user_input_text or None,
                source='telegram',
                telegram_chat_id=update.effective_chat.id,
                telegram_file_id=telegram_file_id,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                status='pending'
            )
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Confirm & Save", callback_data=f"confirm:{pending.id}")],
                [InlineKeyboardButton("✏️ Edit Data", callback_data=f"edit:{pending.id}")]
            ])
            preview_card = await _build_upload_preview_card(
                extracted_json=result,
                confidence=confidence,
                doc_label=display_type,
                file_name=file_name
            )
            await update.message.reply_text(preview_card, reply_markup=keyboard, parse_mode="Markdown")

        latency = time.time() - start_time
        logger.info(f"Document processed for user {db_user.telegram_id} in {latency:.2f}s, awaiting confirmation")

    except Exception:
        logger.exception("Document processing failed for user %s", update.effective_user.id)
        await update.message.reply_text(
            "Sorry — I could not process this document. Please try again with a valid image or PDF."
        )


async def mydocs_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mydocs command to show user's document history."""
    # Check user allowlist
    if not _is_user_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Get or create user in database
    db_user = _get_or_create_user(update)
    if not db_user:
        await update.message.reply_text("Error: Could not identify user.")
        return

    try:
        # Get user's documents
        docs = db_service.get_user_documents(db_user.id, limit=20)

        if not docs:
            await update.message.reply_text(
                "📄 **Your Documents**\n\n"
                "Abhi tak koi document upload nahi hua.\n"
                "Image ya PDF bhejo, main turant extract karke save kar dunga 😊"
            )
            return

        # Build summarized + detailed response
        total_docs = len(docs)
        total_amount = sum((doc.total_amount or 0) for doc in docs)
        image_count = sum(1 for doc in docs if doc.mime_type and doc.mime_type.startswith("image"))
        pdf_count = total_docs - image_count
        avg_amount = (total_amount / total_docs) if total_docs else 0
        vendor_totals = {}
        for doc in docs:
            vendor = doc.vendor_name or "Unknown"
            vendor_totals[vendor] = vendor_totals.get(vendor, 0) + (doc.total_amount or 0)
        top_vendor, top_vendor_amount = max(vendor_totals.items(), key=lambda x: x[1]) if vendor_totals else ("N/A", 0)

        lines = [
            "╔════════════════════╗",
            "📊 **DOCUMENT SNAPSHOT (🆕 v2)**",
            "╚════════════════════╝",
            "",
            f"📄 **Docs:** {total_docs}",
            f"💰 **Total Spend:** ₹{total_amount:,.2f}",
            f"📸 **Images/PDFs:** {image_count}/{pdf_count}",
            f"📈 **Avg per Doc:** ₹{avg_amount:,.2f}",
            f"🏪 **Top Vendor:** {top_vendor}",
            f"   └ ₹{top_vendor_amount:,.2f}",
            "",
            f"🗂️ **Recent {min(total_docs, 20)} Documents**"
        ]

        for idx, doc in enumerate(docs[:20], 1):
            doc_type = "📸" if doc.mime_type and doc.mime_type.startswith("image") else "📑"
            title = doc.title or doc.file_name or "Untitled"
            short_title = title[:36] + "..." if len(title) > 36 else title
            amount = f"₹{doc.total_amount:,.2f}" if doc.total_amount else "N/A"
            date = doc.document_date or "N/A"
            created = doc.created_at.strftime('%d %b %Y, %I:%M %p')
            lines.append(
                f"\n**{idx})** {doc_type} **{short_title}**\n"
                f"🆔 `{doc.id}`   💰 {amount}\n"
                f"📅 {date}   🕒 {created}"
            )

        if total_docs > 20:
            lines.append(f"\n_...and {total_docs - 20} more documents_")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        logger.info(f"User {db_user.telegram_id} viewed {len(docs)} documents")

    except Exception:
        logger.exception("Error fetching documents for user %s", update.effective_user.id)
        await update.message.reply_text("Sorry — I could not fetch your documents. Please try again.")


async def query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /q command for natural language document queries."""
    start_time = time.time()

    # Check user allowlist
    if not _is_user_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Get or create user in database
    db_user = _get_or_create_user(update)
    if not db_user:
        await update.message.reply_text("Error: Could not identify user.")
        return

    # Get the query from command args
    query_text = " ".join(context.args) if context.args else None
    if query_text:
        logger.info("Incoming /q user question: '%s'", query_text)

    if not query_text:
        await update.message.reply_text(
            "📊 **Query Your Documents**\n\n"
            "Ask me anything about your documents using natural language.\n\n"
            "Examples:\n"
            "• `/q summarize my total documents and amount`\n"
            "• `/q show all invoices`\n"
            "• `/q tell me about my receipts`\n"
            "• `/q what vendors do I have?`",
            parse_mode="Markdown"
        )
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        answer = await _answer_user_question(
            update, db_user, query_text, log_prefix="/q"
        )
        await _reply_in_chunks(update, answer, settings.MAX_MESSAGE_LENGTH)

        latency = time.time() - start_time
        logger.info(f"Query processed for user {db_user.telegram_id}: '{query_text}' in {latency:.2f}s")

    except Exception:
        logger.exception("Query processing failed for user %s", update.effective_user.id)
        await update.message.reply_text(
            "Sorry — I could not process your query. Please try again with a different question."
        )


async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summary command to show document summary statistics."""
    start_time = time.time()

    # Check user allowlist
    if not _is_user_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Get or create user in database
    db_user = _get_or_create_user(update)
    if not db_user:
        await update.message.reply_text("Error: Could not identify user.")
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        # Get user summary stats
        stats = db_service.get_user_summary_stats(db_user.id)
        docs = db_service.get_user_documents(db_user.id, limit=20)

        if not docs:
            await update.message.reply_text(
                "📄 **Your Documents**\n\n"
                "Abhi tak koi document upload nahi hua.\n"
                "Image ya PDF bhejo, main turant extract karke save kar dunga 😊"
            )
            return

        # Build summary + detailed response (same style as /mydocs)
        total_docs = stats.get("total_documents", len(docs))
        total_amount = stats.get("total_amount", 0.0)
        image_count = sum(1 for doc in docs if doc.mime_type and doc.mime_type.startswith("image"))
        pdf_count = total_docs - image_count
        avg_amount = (total_amount / total_docs) if total_docs else 0

        vendor_totals = {}
        for doc in docs:
            vendor = doc.vendor_name or "Unknown"
            vendor_totals[vendor] = vendor_totals.get(vendor, 0) + (doc.total_amount or 0)
        top_vendor, top_vendor_amount = max(vendor_totals.items(), key=lambda x: x[1]) if vendor_totals else ("N/A", 0)

        lines = [
            "📊 **Your Document Summary**",
            "",
            (
                f"Aapke paas **{total_docs} documents** hain jinka total amount "
                f"**₹{total_amount:,.2f}** hai. Average per document **₹{avg_amount:,.2f}** "
                f"raha, aur sabse bada vendor contribution **{top_vendor} (₹{top_vendor_amount:,.2f})** ka hai."
            ),
            (
                f"Document mix: **{image_count} images** aur **{pdf_count} PDFs**, "
                f"with **{stats.get('unique_vendors', 0)} unique vendors**."
            ),
        ]

        if stats['document_types']:
            lines.append("\n📁 **Type Breakdown:**")
            for dt in stats['document_types']:
                lines.append(f"• {dt['type']}: {dt['count']}")

        lines.append("")
        lines.append(f"🗂️ **Recent Documents ({min(total_docs, 20)})**")
        for idx, doc in enumerate(docs[:20], 1):
            doc_type = "📸" if doc.mime_type and doc.mime_type.startswith("image") else "📑"
            title = doc.title or doc.file_name or "Untitled"
            short_title = title[:36] + "..." if len(title) > 36 else title
            amount = f"₹{doc.total_amount:,.2f}" if doc.total_amount else "N/A"
            date = doc.document_date or "N/A"
            created = doc.created_at.strftime('%d %b %Y, %I:%M %p')
            lines.append(
                f"{idx}. {doc_type} **{short_title}** — {amount} | {date} | {created}"
            )

        if total_docs > 20:
            lines.append(f"\n_...and {total_docs - 20} more documents_")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

        latency = time.time() - start_time
        logger.info(f"Summary command for user {db_user.telegram_id} completed in {latency:.2f}s")

    except Exception:
        logger.exception("Summary command failed for user %s", update.effective_user.id)


async def confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle confirm button click - save document to database."""
    query = update.callback_query
    await query.answer()

    # Extract pending ID from callback data
    _, pending_id = query.data.split(":", 1)
    pending_id = int(pending_id)

    # Get pending document from database
    pending = db_service.get_pending_document_by_id(pending_id, _get_or_create_user(update).id)

    if not pending:
        await query.edit_message_text("❌ Document not found or expired. Please upload again.")
        return

    try:
        # Confirm and save to database
        doc = db_service.confirm_pending_document(pending_id)

        if not doc:
            await query.edit_message_text("❌ Failed to save document. Please try again.")
            return

        indexed = DatabaseService.index_document_in_vector_store(doc)
        if not indexed:
            indexed = DatabaseService.reindex_document_vector(doc.id, doc.user_id)

        confidence = pending.confidence_overall if pending.confidence_overall is not None else _get_confidence(pending.extracted_data)
        saved_card = await _build_upload_preview_card(
            extracted_json=pending.extracted_data,
            confidence=confidence,
            doc_label="Saved Document",
            file_name=pending.file_name or ""
        )

        vector_note = (
            "\n\n🔎 Indexed in vector DB for semantic search."
            if indexed
            else "\n\n⚠️ Saved in SQL but vector indexing failed — /q may miss this document until re-indexed."
        )

        await query.edit_message_text(
            f"✅ **Document Saved** (SQL + vector)\n\n{saved_card}{vector_note}",
            parse_mode="Markdown"
        )
        _remember_saved_document_for_qa(doc.user_id, pending.extracted_data, pending.file_name or "document")

    except Exception:
        logger.exception("Failed to confirm document")
        await query.edit_message_text("❌ Failed to save document. Please try again.")


async def edit_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle edit button click for pending documents."""
    query = update.callback_query
    await query.answer()

    # Extract pending ID from callback data
    _, pending_id = query.data.split(":", 1)
    pending_id = int(pending_id)

    # Get pending document from database
    pending = db_service.get_pending_document_by_id(pending_id, _get_or_create_user(update).id)

    if not pending:
        await query.edit_message_text("❌ Document not found or expired. Please upload again.")
        return

    # Store in context for reply handler
    context.user_data["editing_pending_id"] = pending_id

    # Ask user to reply with corrected JSON
    await query.edit_message_text(
        f"✏️ **Edit Document Data**\n\n"
        f"Please reply with the corrected JSON data:\n\n"
        f"```json\n{pending.extracted_data}\n```\n\n"
        f"Reply with your corrected JSON to save it.",
        parse_mode="Markdown"
    )


async def edit_reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user reply with edited JSON data."""
    pending_id = context.user_data.get("editing_pending_id")

    if not pending_id:
        return  # Not editing anything

    # Clear the editing state
    del context.user_data["editing_pending_id"]

    try:
        # Validate the JSON
        corrected_json = update.message.text
        corrected_data = json.loads(corrected_json)  # Validate JSON

        # Get pending document from database
        pending = db_service.get_pending_document_by_id(pending_id, _get_or_create_user(update).id)

        if not pending:
            await update.message.reply_text("❌ Document not found or expired. Please upload again.")
            return

        # Update pending document with corrected data
        updated_pending = db_service.update_pending_document(pending_id, corrected_json)

        if not updated_pending:
            await update.message.reply_text("❌ Failed to update document. Please try again.")
            return

        # Confirm and save to database
        doc = db_service.confirm_pending_document(pending_id)

        if not doc:
            await update.message.reply_text("❌ Failed to save document. Please try again.")
            return

        indexed = DatabaseService.index_document_in_vector_store(doc)
        if not indexed:
            indexed = DatabaseService.reindex_document_vector(doc.id, doc.user_id)

        corrected_confidence = _get_confidence(corrected_json)
        corrected_card = await _build_upload_preview_card(
            extracted_json=corrected_json,
            confidence=corrected_confidence,
            doc_label="Corrected Document",
            file_name=pending.file_name or ""
        )

        vector_note = (
            "\n\n🔎 Indexed in vector DB."
            if indexed
            else "\n\n⚠️ SQL saved; vector index failed."
        )

        await update.message.reply_text(
            f"✅ **Corrected Document Saved** (SQL + vector)\n📄 Document ID: `{doc.id}`\n\n{corrected_card}{vector_note}",
            parse_mode="Markdown"
        )
        _remember_saved_document_for_qa(doc.user_id, corrected_json, pending.file_name or "corrected document")

    except json.JSONDecodeError:
        await update.message.reply_text("❌ Invalid JSON format. Please check your input and try again.")
    except Exception:
        logger.exception("Failed to save corrected document")
        await update.message.reply_text("❌ Failed to save document. Please try again.")
