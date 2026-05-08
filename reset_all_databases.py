#!/usr/bin/env python3
"""Reset all databases - SQLite and ChromaDB for fresh start."""

import os
import shutil
from dotenv import load_dotenv

load_dotenv()

# Paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
SQLITE_DB_PATH = "bot_data.db"

def reset_databases():
    """Delete all database data for fresh start."""
    
    # Reset ChromaDB
    if os.path.exists(CHROMA_DB_PATH):
        print(f"🗑️  Deleting ChromaDB at: {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)
        print("✅ ChromaDB deleted")
    else:
        print(f"⚠️  ChromaDB not found at: {CHROMA_DB_PATH}")
    
    # Recreate ChromaDB directory
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    print(f"✅ Created fresh ChromaDB directory")
    
    # Reset SQLite
    if os.path.exists(SQLITE_DB_PATH):
        print(f"🗑️  Deleting SQLite DB at: {SQLITE_DB_PATH}")
        os.remove(SQLITE_DB_PATH)
        print("✅ SQLite DB deleted")
    else:
        print(f"⚠️  SQLite DB not found at: {SQLITE_DB_PATH}")
    
    # Check for other SQLite files
    for ext in ['-shm', '-wal', '-journal']:
        file = f"{SQLITE_DB_PATH}{ext}"
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Deleted {file}")
    
    print("\n🎉 All databases reset successfully!")
    print("📝 Next steps:")
    print("   1. Start the bot")
    print("   2. Upload documents (they'll be embedded with OpenAI 1536-dim)")
    print("   3. Query using natural language")

if __name__ == "__main__":
    reset_databases()
