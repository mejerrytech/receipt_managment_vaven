"""
Migration script: Backfill existing documents from SQLite to ChromaDB.

Usage:
    python migrate_to_chroma.py

This will:
1. Read all documents from SQLite
2. Generate embeddings
3. Store in ChromaDB with user_id guardrails
"""

import os
import sys
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# Import vector service directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.vector_service import get_vector_service


def migrate_documents(batch_size: int = 100):
    """
    Migrate all existing documents from SQLite to ChromaDB.

    Args:
        batch_size: Number of documents to process at once
    """
    # Get database path
    db_path = os.getenv("DATABASE_URL", "sqlite:///bot_data.db").replace("sqlite:///", "")

    # Get vector service
    vector_service = get_vector_service()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Count total documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        total = cursor.fetchone()[0]
        print(f"Found {total} documents to migrate")

        if total == 0:
            print("No documents found. Nothing to migrate.")
            return

        # Process in batches
        offset = 0
        migrated = 0
        failed = 0

        while offset < total:
            # Get batch of documents
            cursor.execute("""
                SELECT id, user_id, file_name, mime_type, document_type, title,
                       total_amount, vendor_name, extracted_data, document_date, raw_text
                FROM documents
                LIMIT ? OFFSET ?
            """, (batch_size, offset))

            rows = cursor.fetchall()

            if not rows:
                break

            for row in rows:
                try:
                    # Prepare document data for vector DB
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
                        "raw_text": row["raw_text"]
                    }

                    # Add to ChromaDB
                    success = vector_service.add_document(
                        doc_id=row["id"],
                        user_id=row["user_id"],
                        doc_data=doc_dict
                    )

                    if success:
                        migrated += 1
                        print(f"✓ Migrated document {row['id']} (User: {row['user_id']})")
                    else:
                        failed += 1
                        print(f"✗ Failed to migrate document {row['id']}")

                except Exception as e:
                    failed += 1
                    print(f"✗ Error migrating document {row['id']}: {e}")

            # Commit batch progress
            print(f"\n--- Progress: {migrated}/{total} migrated, {failed} failed ---\n")
            offset += batch_size

        print(f"\n{'='*50}")
        print(f"Migration complete!")
        print(f"Total documents: {total}")
        print(f"Successfully migrated: {migrated}")
        print(f"Failed: {failed}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Migration failed: {e}")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    print("Starting migration from SQLite to ChromaDB...")
    print(f"ChromaDB path: {os.getenv('CHROMA_DB_PATH', './chroma_db')}")
    print()

    # Confirm before starting
    response = input("This will migrate all existing documents to ChromaDB. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Migration cancelled.")
        sys.exit(0)

    migrate_documents()
