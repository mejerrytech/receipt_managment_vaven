import logging
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from bot_telegram.config import settings
from bot_telegram.handlers import (
    start_handler, help_handler, message_handler,
    websearch_handler, clear_handler, error_handler,
    photo_handler, document_handler, mydocs_handler, query_handler, summary_handler,
    confirm_callback, edit_callback_handler, edit_reply_handler
)

# Setup structured logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def main() -> None:
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    
    logging.info(f"Starting bot in {settings.APP_ENV} mode")
    logging.info(f"Allowed users: {settings.ALLOWED_TELEGRAM_USER_IDS or 'ALL'}")
    logging.info(f"Conversation memory: {settings.CONVERSATION_MEMORY_ENABLED}")
    
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("websearch", websearch_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(CommandHandler("mydocs", mydocs_handler))
    app.add_handler(CommandHandler("q", query_handler))
    app.add_handler(CommandHandler("summary", summary_handler))
    
    # Reply handler for edited JSON (processes first to catch edit replies)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, edit_reply_handler), group=0)

    # Message handler (processes regular messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler), group=1)

    # Photo and document handlers for OCR
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.Document.IMAGE | filters.Document.PDF, document_handler))

    # Callback handlers for edit/confirm flow
    app.add_handler(CallbackQueryHandler(confirm_callback, pattern="^confirm:"))
    app.add_handler(CallbackQueryHandler(edit_callback_handler, pattern="^edit:"))

    # Error handler
    app.add_error_handler(error_handler)
    
    logging.info("Bot started successfully. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
