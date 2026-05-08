#!/usr/bin/env python3
"""Reset ChromaDB collection to fix embedding dimension mismatch."""

import os
import shutil
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

def reset_chroma():
    """Delete ChromaDB data to force fresh start with new embeddings."""
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Deleting ChromaDB at: {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)
        print("✓ ChromaDB deleted successfully")
    else:
        print(f"ChromaDB path not found: {CHROMA_DB_PATH}")

    # Recreate empty directory
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    print(f"✓ Created fresh ChromaDB directory")
    print("\nDone! Next document upload will create collection with OpenAI embeddings (1536 dims)")

if __name__ == "__main__":
    reset_chroma()
