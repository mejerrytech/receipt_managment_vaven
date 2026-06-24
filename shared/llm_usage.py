import logging
import threading
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger("llm_usage")

_lock = threading.Lock()
_totals = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}
_by_provider = defaultdict(lambda: {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
})
_by_model = defaultdict(lambda: {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
})


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def record_usage(
    *,
    provider: str,
    model: str,
    call_type: str,
    input_tokens: Any = None,
    output_tokens: Any = None,
    total_tokens: Any = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    provider = provider or "unknown"
    model = model or "unknown"
    call_type = call_type or "unknown"

    input_count = _safe_int(input_tokens)
    output_count = _safe_int(output_tokens)
    total_count = _safe_int(total_tokens)
    if total_count <= 0:
        total_count = input_count + output_count

    with _lock:
        _totals["calls"] += 1
        _totals["input_tokens"] += input_count
        _totals["output_tokens"] += output_count
        _totals["total_tokens"] += total_count

        provider_totals = _by_provider[provider]
        provider_totals["calls"] += 1
        provider_totals["input_tokens"] += input_count
        provider_totals["output_tokens"] += output_count
        provider_totals["total_tokens"] += total_count

        model_key = f"{provider}:{model}"
        model_totals = _by_model[model_key]
        model_totals["calls"] += 1
        model_totals["input_tokens"] += input_count
        model_totals["output_tokens"] += output_count
        model_totals["total_tokens"] += total_count

        provider_snapshot = dict(provider_totals)
        model_snapshot = dict(model_totals)
        global_snapshot = dict(_totals)

    logger.info(
        "LLM_USAGE provider=%s model=%s type=%s call_tokens[in=%s out=%s total=%s] "
        "provider_total[calls=%s in=%s out=%s total=%s] "
        "model_total[calls=%s in=%s out=%s total=%s] "
        "global_total[calls=%s in=%s out=%s total=%s] details=%s",
        provider,
        model,
        call_type,
        input_count,
        output_count,
        total_count,
        provider_snapshot["calls"],
        provider_snapshot["input_tokens"],
        provider_snapshot["output_tokens"],
        provider_snapshot["total_tokens"],
        model_snapshot["calls"],
        model_snapshot["input_tokens"],
        model_snapshot["output_tokens"],
        model_snapshot["total_tokens"],
        global_snapshot["calls"],
        global_snapshot["input_tokens"],
        global_snapshot["output_tokens"],
        global_snapshot["total_tokens"],
        details or {},
    )


def record_openai_chat(response: Any, *, model: str, call_type: str, details: Optional[dict[str, Any]] = None) -> None:
    usage = getattr(response, "usage", None)
    record_usage(
        provider="openai",
        model=model,
        call_type=call_type,
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        total_tokens=getattr(usage, "total_tokens", 0),
        details=details,
    )


def record_openai_embedding(response: Any, *, model: str, call_type: str, details: Optional[dict[str, Any]] = None) -> None:
    usage = getattr(response, "usage", None)
    record_usage(
        provider="openai",
        model=model,
        call_type=call_type,
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=0,
        total_tokens=getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0),
        details=details,
    )


def record_anthropic_message(response: Any, *, model: str, call_type: str, details: Optional[dict[str, Any]] = None) -> None:
    usage = getattr(response, "usage", None)
    record_usage(
        provider="anthropic",
        model=model,
        call_type=call_type,
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        total_tokens=_safe_int(getattr(usage, "input_tokens", 0)) + _safe_int(getattr(usage, "output_tokens", 0)),
        details=details,
    )


def record_gemini_response(response: Any, *, model: str, call_type: str, details: Optional[dict[str, Any]] = None) -> None:
    usage = getattr(response, "usage_metadata", None)
    record_usage(
        provider="gemini",
        model=model,
        call_type=call_type,
        input_tokens=getattr(usage, "prompt_token_count", 0),
        output_tokens=getattr(usage, "candidates_token_count", 0),
        total_tokens=getattr(usage, "total_token_count", 0),
        details=details,
    )
