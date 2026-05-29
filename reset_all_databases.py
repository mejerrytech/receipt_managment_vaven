#!/usr/bin/env python3
"""Reset PostgreSQL app data and ChromaDB for a fresh start."""

import os
import shutil
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Tables in FK-safe order (children before parents)
PG_TRUNCATE_TABLES = (
    "documents",
    "pending_documents",
    "user_text_entries",
    "users",
    "prompts",
)


def reset_databases():
    """Clear PostgreSQL data and ChromaDB vectors."""
    from shared.database import engine, init_db

    # Reset ChromaDB
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Deleting ChromaDB at: {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)
        print("ChromaDB deleted")
    else:
        print(f"ChromaDB not found at: {CHROMA_DB_PATH}")

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    print("Created fresh ChromaDB directory")

    # Reset PostgreSQL (requires DATABASE_URL)
    table_list = ", ".join(PG_TRUNCATE_TABLES)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"))
    print(f"PostgreSQL tables truncated: {table_list}")

    init_db()
    print("PostgreSQL schema verified and default prompts re-seeded")

    print("\nAll databases reset successfully.")
    print("Next steps:")
    print("  1. Start the bot")
    print("  2. Upload documents (they will be embedded in ChromaDB)")
    print("  3. Query using natural language")


if __name__ == "__main__":
    reset_databases()
