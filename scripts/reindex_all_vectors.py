#!/usr/bin/env python3
"""
Rebuild Chroma embeddings from PostgreSQL for one or all users.

Each user is isolated via user_id (and username) metadata guardrails in Chroma.

Usage:
  # All users (purge old vectors per user, then re-embed from SQL)
  python scripts/reindex_all_vectors.py --all

  # Single user by UUID
  python scripts/reindex_all_vectors.py --user-id c69466ba-2fce-4dff-8c3a-16acae8b25db

  # Single user by Telegram / WhatsApp numeric id
  python scripts/reindex_all_vectors.py --telegram-id 918114437166

  # Dry run — show SQL row counts only
  python scripts/reindex_all_vectors.py --all --dry-run

  # Keep existing Chroma rows and upsert on top (not recommended after UUID migration)
  python scripts/reindex_all_vectors.py --all --no-purge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.env import load_project_dotenv

load_project_dotenv()

from shared.database import DatabaseService, User, get_db, init_db  # noqa: E402
from shared.id_types import as_uuid  # noqa: E402


def _resolve_user(user_id: str | None, telegram_id: int | None) -> User | None:
    if user_id:
        return DatabaseService.get_user_by_id(as_uuid(user_id))
    if telegram_id is not None:
        db = get_db()
        try:
            return db.query(User).filter(User.telegram_id == telegram_id).first()
        finally:
            db.close()
    return None


def _print_stats(stats: dict) -> None:
    if stats.get("error"):
        print(f"  ERROR: {stats['error']} ({stats.get('user_id', '?')})")
        return
    print(
        f"  user={stats.get('username')} id={stats.get('user_id')} "
        f"telegram={stats.get('telegram_id')}"
    )
    print(
        f"    purged={stats.get('purged', 0)} "
        f"docs={stats.get('documents_ok', 0)}/{stats.get('documents_fail', 0)} fail "
        f"text={stats.get('text_entries_ok', 0)}/{stats.get('text_entries_fail', 0)} fail "
        f"pending={stats.get('pending_ok', 0)}/{stats.get('pending_fail', 0)} fail "
        f"total_indexed={stats.get('total_indexed', 0)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Chroma embeddings per user from PostgreSQL"
    )
    parser.add_argument("--all", action="store_true", help="Reindex every user")
    parser.add_argument("--user-id", help="User UUID (users.id)")
    parser.add_argument("--telegram-id", type=int, help="Telegram / WhatsApp numeric id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print SQL row counts; do not touch Chroma",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do not delete existing Chroma rows for the user before reindex",
    )
    parser.add_argument(
        "--skip-pending",
        action="store_true",
        help="Skip pending_documents (only confirmed docs + text entries)",
    )
    args = parser.parse_args()

    if not args.all and not args.user_id and args.telegram_id is None:
        parser.error("Provide --all, --user-id, or --telegram-id")

    init_db()

    include_pending = not args.skip_pending
    purge_first = not args.no_purge

    if args.all:
        db = get_db()
        try:
            users = db.query(User).order_by(User.created_at).all()
        finally:
            db.close()

        if not users:
            print("No users in database.")
            return 0

        print(f"Users to process: {len(users)}")
        if args.dry_run:
            for user in users:
                counts = DatabaseService.count_user_vector_sources(user.id)
                print(
                    f"  {user.username} ({user.id}): "
                    f"documents={counts['documents']} "
                    f"text_entries={counts['text_entries']} "
                    f"pending={counts['pending']}"
                )
            return 0

        totals = {"purged": 0, "indexed": 0, "failed": 0, "users": 0}
        for user in users:
            print(f"\nReindexing {user.username} ({user.id})...")
            stats = DatabaseService.reindex_user_vectors(
                user.id,
                include_pending=include_pending,
                purge_first=purge_first,
            )
            _print_stats(stats)
            if not stats.get("error"):
                totals["users"] += 1
                totals["purged"] += stats.get("purged", 0)
                totals["indexed"] += stats.get("total_indexed", 0)
                totals["failed"] += stats.get("total_failed", 0)

        print(
            f"\nDone. users={totals['users']} purged={totals['purged']} "
            f"indexed={totals['indexed']} failed={totals['failed']}"
        )
        return 0 if totals["failed"] == 0 else 2

    user = _resolve_user(args.user_id, args.telegram_id)
    if not user:
        print("User not found", file=sys.stderr)
        return 1

    if args.dry_run:
        counts = DatabaseService.count_user_vector_sources(user.id)
        print(json.dumps({"user": user.to_dict(), "sources": counts}, indent=2))
        return 0

    stats = DatabaseService.reindex_user_vectors(
        user.id,
        include_pending=include_pending,
        purge_first=purge_first,
    )
    print(f"Reindexed user {user.id} ({user.username}):")
    _print_stats(stats)
    return 0 if not stats.get("error") and stats.get("total_failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
