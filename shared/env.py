"""Load .env from the project tree (works regardless of process cwd)."""

from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv() -> Path | None:
    """Find .env next to this package or in parent dirs; override stale shell vars."""
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        env_path = directory / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=True)
            return env_path
    load_dotenv(override=True)
    return None
