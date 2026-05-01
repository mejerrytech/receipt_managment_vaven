import io
import json
import logging
import time
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes
from shared.openai_client import OpenAIService
from shared.database import DatabaseService
from shared.nlp_sql_service import get_nlp_sql_service
from shared.ocr_service import get_ocr_service
from bot_telegram.config import settings

logger = logging.getLogger("telegram_handlers")
openai_service = OpenAIService()
db_service = DatabaseService()
ocr_service = get_ocr_service()


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
        "• Web search (/websearch)\n"
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
        "/websearch <query> - Search the web with AI\n"
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
    await update.message.reply_text("Conversation memory cleared!")


async def websearch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /websearch command to search the web."""
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
    chat_id = str(update.effective_chat.id)
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    try:
        # Use conversation memory if enabled
        use_memory = settings.CONVERSATION_MEMORY_ENABLED
        
        answer = await openai_service.ask(
            user_prompt=user_text,
            system_prompt=settings.SYSTEM_PROMPT,
            chat_id=chat_id if use_memory else None,
            use_memory=use_memory
        )
        
        # Send response (chunked if too long)
        await _reply_in_chunks(update, answer, settings.MAX_MESSAGE_LENGTH)
        
        latency = time.time() - start_time
        logger.info(
            f"Message processed for user {update.effective_user.id} "
            f"in {latency:.2f}s (memory: {use_memory})"
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
        file = await photo.get_file()

        # Download image bytes
        image_bytes = await file.download_as_bytearray()
        file_size = len(image_bytes)

        logger.info(f"Processing photo from user {db_user.telegram_id} (@{db_user.username}), size: {file_size} bytes")

        # Extract data using OCR (Gemini 2.5 Flash + Claude Opus 4.5)
        result = await ocr_service.extract_data(
            image_bytes=bytes(image_bytes),
            mime_type="image/jpeg"
        )

        # Check confidence score
        confidence = _get_confidence(result)

        # Store pending document in database
        pending = db_service.create_pending_document(
            user_id=db_user.id,
            file_name="photo.jpg",
            mime_type="image/jpeg",
            file_size=file_size,
            extracted_json=result,
            confidence_overall=confidence,
            source='telegram',
            telegram_chat_id=update.effective_chat.id
        )

        # Show data with edit/confirm buttons
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Confirm & Save", callback_data=f"confirm:{pending.id}")],
            [InlineKeyboardButton("✏️ Edit Data", callback_data=f"edit:{pending.id}")]
        ])

        if confidence < 0.8:
            header = f"⚠️ **Low Confidence: {confidence:.0%}**\n📸 **Image Analysis**\n\n"
        else:
            header = f"📸 **Image Analysis**\n✅ **Confidence: {confidence:.0%}**\n\n"
        
        await update.message.reply_text(
            header + f"```json\n{result}\n```",
            reply_markup=keyboard,
            parse_mode="Markdown"
        )

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
        file_bytes = await file.download_as_bytearray()
        file_size = len(file_bytes)

        logger.info(f"Processing document from user {db_user.telegram_id} (@{db_user.username}): {file_name}, size: {file_size} bytes")

        # For PDFs, note that GPT-4 Vision works best with image-based PDFs
        display_type = "PDF" if mime_type == "application/pdf" else "Image"

        # Extract data using OCR (Gemini 2.5 Flash + Claude Opus 4.5)
        result = await ocr_service.extract_data(
            image_bytes=bytes(file_bytes),
            mime_type=mime_type
        )

        # Check confidence score
        confidence = _get_confidence(result)

        # Store pending document in database
        pending = db_service.create_pending_document(
            user_id=db_user.id,
            file_name=file_name,
            mime_type=mime_type,
            file_size=file_size,
            extracted_json=result,
            confidence_overall=confidence,
            source='telegram',
            telegram_chat_id=update.effective_chat.id
        )

        # Show data with edit/confirm buttons
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Confirm & Save", callback_data=f"confirm:{pending.id}")],
            [InlineKeyboardButton("✏️ Edit Data", callback_data=f"edit:{pending.id}")]
        ])

        if confidence < 0.8:
            header = f"⚠️ **Low Confidence: {confidence:.0%}**\n📄 **{display_type} Analysis**\n_File: {file_name}_\n\n"
        else:
            header = f"📄 **{display_type} Analysis**\n✅ **Confidence: {confidence:.0%}**\n_File: {file_name}_\n\n"
        
        await update.message.reply_text(
            header + f"```json\n{result}\n```",
            reply_markup=keyboard,
            parse_mode="Markdown"
        )

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
                "📄 **Your Documents**\n\nYou haven't uploaded any documents yet.\n"
                "Send me an image or PDF to extract data!"
            )
            return

        # Build response
        lines = ["📄 **Your Documents**\n"]
        for doc in docs:
            doc_type = "📸" if doc.mime_type and doc.mime_type.startswith("image") else "📄"
            title = doc.title or doc.file_name or "Untitled"
            amount = f"₹{doc.total_amount:.2f}" if doc.total_amount else "N/A"
            date = doc.document_date or "N/A"
            lines.append(
                f"{doc_type} **ID: {doc.id}** | {title[:30]}{'...' if len(title) > 30 else ''}\n"
                f"   Amount: {amount} | Date: {date} | {doc.created_at.strftime('%Y-%m-%d %H:%M')}\n"
            )

        lines.append(f"\n_Total: {len(docs)} documents_")

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
        # Use NLP SQL service which has intent routing built-in
        nlp_service = get_nlp_sql_service()
        result = nlp_service.ask_ai(query_text, db_user.id)

        # Format response based on result
        if result.get("success"):
            # Check if it's a fallback/semantic search result
            if result.get("fallback"):
                # Semantic search results - show TOP 3 matches with ALL fields
                data = result.get("data", [])
                
                if data:
                    all_lines = []
                    
                    # Show top 3 results
                    for idx, item in enumerate(data[:3], 1):
                        lines = [f"{idx}. **{item.get('title', 'Document')}**"]
                        
                        # Define field icons and display order (aliases mapped to single fields)
                        field_map = {
                            'type': ('📄', 'Type'),  # Will also check document_type as alias
                            'amount': ('💰', 'Amount'),  # Will also check total_amount as alias
                            'vendor': ('🏪', 'Vendor'),  # Will also check vendor_name as alias
                            'date': ('📅', 'Date'),  # Will also check document_date, created_at as aliases
                            'currency': ('💵', 'Currency'),
                            'invoice_number': ('🔢', 'Invoice #'),
                            'gstin': ('🆔', 'GSTIN'),
                            'file_name': ('📁', 'File'),
                        }
                        
                        # Track which fields we've displayed (including aliases)
                        displayed = set()
                        
                        # Show fields in preferred order with alias handling
                        # Define all aliases to prevent duplicates
                        all_aliases = {
                            'type': ['type', 'document_type'],
                            'amount': ['amount', 'total_amount'],
                            'vendor': ['vendor', 'vendor_name'],
                            'date': ['date', 'document_date', 'created_at']
                        }
                        
                        for key, (icon, label) in field_map.items():
                            val = None
                            keys_to_check = all_aliases.get(key, [key])
                            for k in keys_to_check:
                                if k in item and item[k] is not None and k not in displayed:
                                    val = item[k]
                                    # Mark ALL aliases as displayed to prevent duplicates
                                    for alias in keys_to_check:
                                        displayed.add(alias)
                                    break
                            
                            if val is not None:
                                # Format amounts with currency (handle None currency)
                                if key == 'amount' and isinstance(val, (int, float)):
                                    currency = item.get('currency') or '₹'
                                    val = f"{currency}{val:.2f}"
                                lines.append(f"   {icon} {label}: {val}")
                        
                        # Show any remaining fields (not aliases of already shown)
                        skip_fields = {'_text', '_score', 'title', 'user_id', 'id', 
                                       'type', 'document_type', 'amount', 'total_amount',
                                       'vendor', 'vendor_name', 'date', 'document_date', 'created_at'}
                        for key, val in item.items():
                            if key not in skip_fields and key not in displayed and val is not None:
                                if not key.startswith('_'):
                                    lines.append(f"   • {key}: {val}")
                        
                        all_lines.append("\n".join(lines))
                    
                    if len(data) > 3:
                        all_lines.append(f"\n_... and {len(data) - 3} more results_")

                    await update.message.reply_text("\n\n".join(all_lines), parse_mode="Markdown")
                else:
                    await update.message.reply_text(
                        "🔍 No matching documents found for your query.\n"
                        "Try asking about specific topics or upload more documents!"
                    )
            else:
                # SQL query results - use LLM-generated summary
                ai_response = result.get("ai_response")
                if ai_response:
                    await update.message.reply_text(ai_response, parse_mode="Markdown")
                else:
                    await update.message.reply_text("📊 Query executed successfully but no summary available.")
        else:
            # Error or no results
            ai_response = result.get("ai_response", "I could not process your query. Please try rephrasing it.")
            await update.message.reply_text(f"⚠️ {ai_response}")

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

        # Build formatted response
        lines = ["📊 **Your Document Summary**\n"]
        lines.append(f"📄 **Total Documents:** {stats['total_documents']}")
        lines.append(f"💰 **Total Amount:** ₹{stats['total_amount']:.2f}")
        lines.append(f"🏪 **Unique Vendors:** {stats['unique_vendors']}")

        if stats['document_types']:
            lines.append("\n📁 **Document Types:**")
            for dt in stats['document_types']:
                lines.append(f"  • {dt['type']}: {dt['count']}")

        if stats['recent_documents']:
            lines.append("\n🕒 **Recent Documents:**")
            for i, doc in enumerate(stats['recent_documents'][:5], 1):
                amount_str = f" (₹{doc['amount']:.2f})" if doc['amount'] else ""
                lines.append(f"  {i}. {doc['title']}{amount_str}")

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

        # Update message
        await query.edit_message_text(
            f"✅ **Document Saved**\n📄 Document ID: `{doc.id}`\n\n"
            f"```json\n{pending.extracted_data}\n```",
            parse_mode="Markdown"
        )

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

        await update.message.reply_text(
            f"✅ **Corrected Document Saved**\n📄 Document ID: `{doc.id}`\n\n"
            f"```json\n{corrected_json}\n```",
            parse_mode="Markdown"
        )

    except json.JSONDecodeError:
        await update.message.reply_text("❌ Invalid JSON format. Please check your input and try again.")
    except Exception:
        logger.exception("Failed to save corrected document")
        await update.message.reply_text("❌ Failed to save document. Please try again.")
