from __future__ import annotations

from uuid import UUID

import json
import uuid
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Optional

from api.reports.constants import REPORTS_DIR


def _user_dir(user_id: UUID) -> Path:
    path = REPORTS_DIR / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _history_path(user_id: UUID) -> Path:
    return _user_dir(user_id) / "history.json"


def _settings_path(user_id: UUID) -> Path:
    return _user_dir(user_id) / "auto_send.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def report_file_path(user_id: UUID, report_id: str, ext: str) -> Path:
    return _user_dir(user_id) / f"{report_id}.{ext}"


def add_history_record(user_id: UUID, record: dict[str, Any]) -> dict[str, Any]:
    path = _history_path(user_id)
    history: list[dict[str, Any]] = _load_json(path, [])
    history.insert(0, record)
    _save_json(path, history)
    return record


def list_history(user_id: UUID, page: int = 1, limit: int = 20) -> dict[str, Any]:
    history: list[dict[str, Any]] = _load_json(_history_path(user_id), [])
    total = len(history)
    total_pages = ceil(total / limit) if total else 0
    offset = (page - 1) * limit
    items = history[offset : offset + limit]
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }


def get_history_record(user_id: UUID, report_id: str) -> Optional[dict[str, Any]]:
    history: list[dict[str, Any]] = _load_json(_history_path(user_id), [])
    for item in history:
        if item.get("report_id") == report_id:
            return item
    return None


def get_latest_report(user_id: UUID) -> Optional[dict[str, Any]]:
    history: list[dict[str, Any]] = _load_json(_history_path(user_id), [])
    return history[0] if history else None


def new_report_id() -> str:
    return uuid.uuid4().hex


def get_auto_send_settings(user_id: UUID) -> dict[str, Any]:
    default = {"enabled": False, "ca_name": None, "ca_email": None, "schedule": "monthly"}
    return {**default, **_load_json(_settings_path(user_id), {})}


def save_auto_send_settings(user_id: UUID, settings: dict[str, Any]) -> dict[str, Any]:
    current = get_auto_send_settings(user_id)
    current.update(settings)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(_settings_path(user_id), current)
    return current
