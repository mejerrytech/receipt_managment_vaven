import os
import re
from typing import Set

from shared.env import load_project_dotenv

load_project_dotenv()


def normalize_whatsapp_number(value: str | None) -> str:
    """Normalize Twilio WhatsApp addresses to digits-only phone numbers."""
    if not value:
        return ""
    cleaned = value.strip().lower().replace("whatsapp:", "")
    return re.sub(r"\D", "", cleaned)


class Settings:
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    TWILIO_DEFAULT_CONTENT_SID = os.getenv("TWILIO_DEFAULT_CONTENT_SID", "")

    APP_ENV = os.getenv("APP_ENV", "whatsapp-dev")
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "3500"))
    WHATSAPP_MEDIA_TIMEOUT_SECONDS = int(os.getenv("WHATSAPP_MEDIA_TIMEOUT_SECONDS", "30"))
    WHATSAPP_MAX_BODY_LENGTH = int(os.getenv("WHATSAPP_MAX_BODY_LENGTH", "4000"))
    WHATSAPP_MAX_MEDIA_COUNT = int(os.getenv("WHATSAPP_MAX_MEDIA_COUNT", "1"))
    WHATSAPP_VERIFY_TWILIO_SIGNATURE = os.getenv("WHATSAPP_VERIFY_TWILIO_SIGNATURE", "false").lower() == "true"
    WHATSAPP_PUBLIC_BASE_URL = os.getenv("WHATSAPP_PUBLIC_BASE_URL", "").rstrip("/")
    WHATSAPP_MESSAGE_CACHE_SECONDS = int(os.getenv("WHATSAPP_MESSAGE_CACHE_SECONDS", "900"))

    ALLOWED_WHATSAPP_NUMBERS: Set[str] = set(
        normalize_whatsapp_number(num)
        for num in os.getenv("ALLOWED_WHATSAPP_NUMBERS", "").split(",")
        if normalize_whatsapp_number(num)
    )

    @classmethod
    def is_number_allowed(cls, number: str) -> bool:
        if not cls.ALLOWED_WHATSAPP_NUMBERS:
            return True
        return normalize_whatsapp_number(number) in cls.ALLOWED_WHATSAPP_NUMBERS


settings = Settings()
