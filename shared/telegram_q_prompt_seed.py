"""Load default Telegram `/q` prompt rows for DB seeding (see telegram_q_prompt_seed.json)."""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_telegram_q_seed_rows() -> List[Dict[str, Any]]:
    path = Path(__file__).with_name("telegram_q_prompt_seed.json")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
