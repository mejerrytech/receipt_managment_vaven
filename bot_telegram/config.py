import os
from dotenv import load_dotenv
from typing import Set

# Load environment variables from .env file
load_dotenv()

class Settings:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    APP_ENV = os.getenv("APP_ENV", "telegram-dev")
    SYSTEM_PROMPT = (
        "You are a helpful Telegram assistant connected to a product that can store receipts and track spending — "
        "but the user may send anything: greetings, general chat, how-to questions, unrelated topics, or "
        "receipt/expense questions. Always infer intention from the actual message; do not treat every turn as "
        "expense-only. "
        "Be concise, accurate, and action-oriented. "
        "Match the user's language and script (e.g. Hindi, Hinglish, English) from their latest message. "
        "When they are clearly logging spend in free text vs asking about saved data or totals, behave "
        "accordingly; for data questions, use conversation context or say you do not know — never invent saves. "
        "Keep replies short for mobile chat."
    )

    # User allowlist - comma-separated list of allowed Telegram user IDs
    # If empty, all users are allowed (for development)
    ALLOWED_TELEGRAM_USER_IDS: Set[str] = set(
        uid.strip() for uid in os.getenv("ALLOWED_TELEGRAM_USER_IDS", "").split(",")
        if uid.strip()
    )
    
    # OpenAI settings
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "45"))  # seconds
    OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    
    # Response chunking
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "3500"))
    
    # Conversation memory
    CONVERSATION_MEMORY_ENABLED = os.getenv("CONVERSATION_MEMORY_ENABLED", "true").lower() == "true"
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    @classmethod
    def is_user_allowed(cls, user_id: str) -> bool:
        """Check if user is in allowlist. If allowlist is empty, allow all."""
        if not cls.ALLOWED_TELEGRAM_USER_IDS:
            return True  # Allow all if no restrictions set
        return user_id in cls.ALLOWED_TELEGRAM_USER_IDS

settings = Settings()
