import json
import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as redis  # pragma: no cover
    RedisClient = redis.Redis
else:
    RedisClient = Any


@dataclass(frozen=True)
class WhatsappRequestRef:
    phone_number: str
    request_id: str

    @property
    def key(self) -> str:
        return f"expense:{self.phone_number}:{self.request_id}"

    @property
    def sent_key(self) -> str:
        return f"{self.key}:sent"


def _now_ts() -> float:
    return time.time()


async def get_redis_client(redis_url: str) -> RedisClient:
    try:
        import redis.asyncio as redis  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Redis python package is not installed in this environment. "
            "Activate the correct venv and run: pip install redis"
        ) from e

    client = redis.from_url(redis_url, decode_responses=True)
    # Ping lazily to fail fast on first use.
    await client.ping()
    return client


async def init_request_state(
    client: RedisClient,
    ref: WhatsappRequestRef,
    *,
    ttl_seconds: int,
    from_address: str,
    message_sid: str,
) -> None:
    payload = {
        "status": "processing",
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
        "from": from_address,
        "message_sid": message_sid,
        "ocr": None,
        "extraction": None,
        "categorization": None,
        "final_message": None,
        "error": None,
    }
    await client.set(ref.key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds, nx=True)
    await client.expire(ref.key, ttl_seconds)


async def load_state(client: RedisClient, ref: WhatsappRequestRef) -> dict[str, Any]:
    raw = await client.get(ref.key)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


async def patch_state(
    client: RedisClient,
    ref: WhatsappRequestRef,
    patch: dict[str, Any],
    *,
    ttl_seconds: int,
) -> None:
    state = await load_state(client, ref)
    if not state:
        state = {"created_at": _now_ts()}
    state.update(patch)
    state["updated_at"] = _now_ts()
    await client.set(ref.key, json.dumps(state, ensure_ascii=False), ex=ttl_seconds)


async def mark_error(
    client: RedisClient,
    ref: WhatsappRequestRef,
    message: str,
    *,
    ttl_seconds: int,
) -> None:
    await patch_state(
        client,
        ref,
        {
            "status": "completed",
            "error": (message or "").strip() or "Unknown error",
        },
        ttl_seconds=ttl_seconds,
    )


async def mark_completed(
    client: RedisClient,
    ref: WhatsappRequestRef,
    *,
    final_message: str,
    ttl_seconds: int,
) -> None:
    await patch_state(
        client,
        ref,
        {
            "status": "completed",
            "final_message": (final_message or "").strip(),
            "error": None,
        },
        ttl_seconds=ttl_seconds,
    )


async def claim_process_once(
    client: RedisClient,
    ref: WhatsappRequestRef,
    *,
    ttl_seconds: int,
) -> bool:
    """Ensure only one worker runs OCR/card pipeline for this request."""
    ok = await client.set(f"{ref.key}:processing", "1", nx=True, ex=ttl_seconds)
    return bool(ok)


async def claim_send_once(
    client: RedisClient,
    ref: WhatsappRequestRef,
    *,
    ttl_seconds: int,
) -> bool:
    # Guard to ensure we only send once even under retries.
    ok = await client.set(ref.sent_key, "1", nx=True, ex=ttl_seconds)
    return bool(ok)


async def cleanup_request_state(client: RedisClient, ref: WhatsappRequestRef) -> None:
    await client.delete(ref.key)
    await client.delete(ref.sent_key)
    await client.delete(f"{ref.key}:processing")


def extract_basic_fields_from_ocr_json(ocr_json: str) -> dict[str, Optional[str]]:
    """
    Best-effort extraction for the WhatsApp final summary.
    We keep it tolerant to schema differences (Gemini output variants).
    """
    try:
        data = json.loads(ocr_json) if ocr_json else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    vendor_name: Optional[str] = None
    vendor = data.get("vendor_or_sender")
    if isinstance(vendor, dict):
        vendor_name = (vendor.get("name") or "").strip() or None
    if not vendor_name and isinstance(data.get("vendor"), str):
        vendor_name = data.get("vendor").strip() or None

    amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
    total = amounts.get("total")
    if total is None:
        total = data.get("total_amount")
    amount_str = None
    if total is not None:
        try:
            amount_str = str(total).strip()
        except Exception:
            amount_str = None

    currency = (amounts.get("currency") or data.get("currency") or "INR")
    date = (data.get("date") or "").strip() or None
    category = (data.get("expense_category") or "").strip() or None

    return {
        "merchant": vendor_name,
        "amount": amount_str,
        "currency": str(currency).strip() if currency else None,
        "date": date,
        "category": category,
    }

