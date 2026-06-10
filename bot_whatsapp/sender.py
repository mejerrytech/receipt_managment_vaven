import json
from typing import Any, Dict, Optional

import requests

from bot_whatsapp.config import normalize_whatsapp_number, settings


def _messages_url() -> str:
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are required")
    return f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json"


def _post_message(data: Dict[str, str]) -> Dict[str, Any]:
    response = requests.post(
        _messages_url(),
        data=data,
        auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def send_whatsapp_message(to_number: str, body: str):
    """Send a free-form WhatsApp message through Twilio."""
    normalized_to = normalize_whatsapp_number(to_number)
    return _post_message({
        "From": settings.TWILIO_WHATSAPP_FROM,
        "Body": body,
        "To": f"whatsapp:+{normalized_to}",
    })


def send_whatsapp_template(
    to_number: str,
    content_variables: Dict[str, Any],
    content_sid: Optional[str] = None,
):
    """Send a Twilio Content Template message."""
    sid = content_sid or settings.TWILIO_DEFAULT_CONTENT_SID
    if not sid:
        raise RuntimeError("content_sid or TWILIO_DEFAULT_CONTENT_SID is required")
    normalized_to = normalize_whatsapp_number(to_number)
    return _post_message({
        "From": settings.TWILIO_WHATSAPP_FROM,
        "ContentSid": sid,
        "ContentVariables": json.dumps(content_variables),
        "To": f"whatsapp:+{normalized_to}",
    })
