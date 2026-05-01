import os
from dotenv import load_dotenv
from typing import Set

# Load environment variables from .env file
load_dotenv()

class Settings:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    APP_ENV = os.getenv("APP_ENV", "telegram-dev")
    SYSTEM_PROMPT = (
        "You are a helpful productivity assistant running in Telegram. "
        "Be concise, accurate, and action-oriented."
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
