"""Lightweight heuristics to skip expensive LLM calls on obvious messages."""

from __future__ import annotations

import re
from typing import Optional

_GREETING_RE = re.compile(
    r"(?i)^\s*("
    r"hi+|hii+|hey+|hello+|hola+|namaste|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"gm|gn|"
    r"thanks?|thank\s*you|thx|"
    r"ok(?:ay)?|bye+|goodbye|see\s*ya"
    r")\s*[!.,?🙂😊👋]*\s*$"
)

_THANKS_RE = re.compile(r"(?i)^\s*(thanks?|thank\s*you|thx|dhanyavaad|shukriya)\s*[!.,?]*\s*$")

_AMOUNT_RE = re.compile(
    r"(?i)(?:₹|rs\.?|inr)\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:₹|rs\.?|inr)"
)
_EXPENSE_HINT_RE = re.compile(
    r"(?i)\b("
    r"paid|spend|spent|kharch|buy|bought|purchase|liya|liye|dalwaya|"
    r"bill|rent|petrol|diesel|fuel|milk|grocery|shopping|expense|payment|"
    r"udhaar|borrowed|loan|emi|subscription|recharge|order"
    r")\b"
)


def is_simple_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match((text or "").strip()))


def is_simple_thanks(text: str) -> bool:
    return bool(_THANKS_RE.match((text or "").strip()))


def looks_like_expense_message(text: str) -> bool:
    """True when message likely logs money spent (not a document question)."""
    cleaned = (text or "").strip()
    if len(cleaned) < 4:
        return False
    if is_simple_greeting(cleaned) or is_simple_thanks(cleaned):
        return False
    has_amount = bool(_AMOUNT_RE.search(cleaned)) or bool(re.search(r"\b\d{2,}\b", cleaned))
    has_hint = bool(_EXPENSE_HINT_RE.search(cleaned))
    return has_amount and (has_hint or len(cleaned) < 120)


def looks_like_multi_item_expense(text: str) -> bool:
    """True when message likely contains multiple purchases in one paragraph."""
    cleaned = (text or "").strip()
    if len(cleaned) < 25:
        return False
    amounts = re.findall(r"\b[\d,]{2,}(?:\.\d+)?\b", cleaned)
    if len(amounts) >= 2:
        return True
    if len(amounts) >= 1 and cleaned.count(".") >= 2:
        return True
    if len(amounts) >= 1 and re.search(r"(?i)\b(aur|and|also|phir|fir)\b", cleaned):
        return len(cleaned.split()) >= 8
    return False


def greeting_reply(text: str) -> Optional[str]:
    if is_simple_thanks(text):
        return "You're welcome! Aur receipts ya expenses ke liye help chahiye ho to bataiye."
    if is_simple_greeting(text):
        return (
            "Hi! Main aapki receipts aur expenses me help kar sakta hoon.\n"
            "Image/PDF bhejein ya documents ke bare me question puchhein."
        )
    return None
