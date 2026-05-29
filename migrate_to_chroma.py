"""
Migration script: Backfill existing documents from PostgreSQL to ChromaDB.

Usage:
    python migrate_to_chroma.py

This will:
1. Read all documents from PostgreSQL (DATABASE_URL)
2. Generate embeddings
3. Store in ChromaDB with user_id guardrails
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.database import engine
from shared.vector_service import get_vector_service


def migrate_documents(batch_size: int = 100):
    """Migrate all existing documents from PostgreSQL to ChromaDB."""
    vector_service = get_vector_service()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
        print(f"Found {total} documents to migrate")

        if total == 0:
            print("No documents found. Nothing to migrate.")
            return

        offset = 0
        migrated = 0
        failed = 0

        while offset < total:
            rows = conn.execute(
                text(
                    """
                    SELECT id, user_id, file_name, mime_type, document_type, title,
                           total_amount, vendor_name, extracted_data, document_date, raw_text
                    FROM documents
                    ORDER BY id
                    LIMIT :limit OFFSET :offset
                    """
                ),
                {"limit": batch_size, "offset": offset},
            ).mappings().all()

            if not rows:
                break

            for row in rows:
                try:
                    doc_dict = {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "file_name": row["file_name"],
                        "document_type": row["document_type"],
                        "title": row["title"],
                        "vendor_name": row["vendor_name"],
                        "total_amount": row["total_amount"],
                        "extracted_json": row["extracted_data"],
                        "document_date": row["document_date"],
                        "raw_text": row["raw_text"],
                    }

                    success = vector_service.add_document(
                        doc_id=row["id"],
                        user_id=row["user_id"],
                        doc_data=doc_dict,
                    )

                    if success:
                        migrated += 1
                        print(f"Migrated document {row['id']} (User: {row['user_id']})")
                    else:
                        failed += 1
                        print(f"Failed to migrate document {row['id']}")

                except Exception as e:
                    failed += 1
                    print(f"Error migrating document {row['id']}: {e}")

            print(f"\n--- Progress: {migrated}/{total} migrated, {failed} failed ---\n")
            offset += batch_size

        print(f"\n{'=' * 50}")
        print("Migration complete!")
        print(f"Total documents: {total}")
        print(f"Successfully migrated: {migrated}")
        print(f"Failed: {failed}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    print("Starting migration from PostgreSQL to ChromaDB...")
    print(f"DATABASE_URL: {os.getenv('DATABASE_URL', '(not set)')}")
    print(f"ChromaDB path: {os.getenv('CHROMA_DB_PATH', './chroma_db')}")
    print()

    response = input("This will migrate all existing documents to ChromaDB. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Migration cancelled.")
        sys.exit(0)

    migrate_documents()
