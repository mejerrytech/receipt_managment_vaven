import os
import json
import logging
import secrets
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, inspect, Column, Integer, String, DateTime, Text, ForeignKey, BigInteger, Float, Boolean, text, or_
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func

from shared.id_types import as_str

from shared.env import load_project_dotenv

load_project_dotenv()

logger = logging.getLogger("database")

# Import vector service for ChromaDB integration
from shared.vector_service import get_vector_service, VectorService

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required. Set it in .env (e.g. New/.env) to a PostgreSQL URL."
    )
if DATABASE_URL.strip().lower().startswith("sqlite"):
    raise RuntimeError(
        "SQLite is not supported. Set DATABASE_URL to PostgreSQL "
        "(e.g. postgresql+psycopg2://user:pass@localhost:5432/expence)."
    )

# Create engine
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def _uuid_pk():
    return Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _uuid_fk(column: str = "users.id"):
    return Column(PG_UUID(as_uuid=True), ForeignKey(column), nullable=False, index=True)


def _timestamp_column_type_sql() -> str:
    return "TIMESTAMP WITH TIME ZONE"


def _table_column_names(table_name: str) -> set:
    """Return lowercase column names via SQLAlchemy inspector."""
    try:
        insp = inspect(engine)
        if not insp.has_table(table_name):
            return set()
        return {c["name"].lower() for c in insp.get_columns(table_name)}
    except Exception as e:
        logger.warning("Could not introspect table %s: %s", table_name, e)
        return set()


def _column_is_integer_pk(table_name: str) -> bool:
    try:
        insp = inspect(engine)
        if not insp.has_table(table_name):
            return False
        for col in insp.get_columns(table_name):
            if col["name"].lower() == "id":
                return "INT" in str(col["type"]).upper()
        return False
    except Exception as e:
        logger.warning("Could not inspect PK type for %s: %s", table_name, e)
        return False


# Legacy migration used predictable UUIDs (e.g. 00000000-0000-4000-8000-000000000002).
# Detect rows that still use that scheme so we can replace them with random UUIDs.
_LEGACY_UUID_RE = r"^[0-9a-f]{8}-0000-4000-8000-[0-9a-f]{12}$"
_LEGACY_UUID_NAMESPACES = {
    "documents": "10000000",
    "pending_documents": "20000000",
    "user_text_entries": "30000000",
    "prompts": "40000000",
}


def _has_legacy_deterministic_uuids(conn) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1 FROM users
            WHERE id::text ~ :pat
            LIMIT 1
            """
        ),
        {"pat": _LEGACY_UUID_RE},
    ).first()
    return row is not None


def _rerandomize_deterministic_uuids() -> None:
    """Replace predictable migration UUIDs with gen_random_uuid() values."""
    with engine.connect() as conn:
        if not _has_legacy_deterministic_uuids(conn):
            return

    logger.warning("Replacing legacy deterministic UUIDs with random UUIDs")
    child_tables = ("documents", "pending_documents", "user_text_entries")

    with engine.begin() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))

        conn.execute(
            text(
                """
                CREATE TEMP TABLE _user_uuid_map AS
                SELECT id AS old_id, gen_random_uuid() AS new_id
                FROM users
                WHERE id::text ~ :pat
                """
            ),
            {"pat": _LEGACY_UUID_RE},
        )

        for child in child_tables:
            _drop_table_fk_constraints(conn, child)

        conn.execute(
            text(
                """
                UPDATE users u
                SET id = m.new_id
                FROM _user_uuid_map m
                WHERE u.id = m.old_id
                """
            )
        )
        for child in child_tables:
            conn.execute(
                text(
                    f"""
                    UPDATE {child} c
                    SET user_id = m.new_id
                    FROM _user_uuid_map m
                    WHERE c.user_id = m.old_id
                    """
                )
            )

        for table, namespace in _LEGACY_UUID_NAMESPACES.items():
            if not inspect(engine).has_table(table):
                continue
            conn.execute(
                text(
                    f"""
                    UPDATE {table}
                    SET id = gen_random_uuid()
                    WHERE id::text LIKE :prefix
                    """
                ),
                {"prefix": f"{namespace}-0000-4000-8000-%"},
            )

        for child in child_tables:
            conn.execute(
                text(
                    f"ALTER TABLE {child} ADD CONSTRAINT {child}_user_id_fkey "
                    f"FOREIGN KEY (user_id) REFERENCES users(id)"
                )
            )

    logger.info("Legacy deterministic UUIDs replaced with random values")


def _column_is_uuid_pk(table_name: str) -> bool:
    try:
        insp = inspect(engine)
        if not insp.has_table(table_name):
            return False
        for col in insp.get_columns(table_name):
            if col["name"].lower() == "id":
                return "UUID" in str(col["type"]).upper()
        return False
    except Exception as e:
        logger.warning("Could not inspect PK type for %s: %s", table_name, e)
        return False


def _cleanup_partial_uuid_migration(conn) -> None:
    """Remove leftover columns from an older add-column migration attempt."""
    for table in ("users", "documents", "pending_documents", "user_text_entries", "prompts"):
        cols = _table_column_names(table)
        if "id_new" in cols:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS id_new"))
        if table != "users" and "user_id_new" in cols:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS user_id_new"))


def _drop_table_fk_constraints(conn, table_name: str) -> None:
    rows = conn.execute(
        text(
            """
            SELECT conname FROM pg_constraint
            WHERE conrelid = CAST(:tbl AS regclass) AND contype = 'f'
            """
        ),
        {"tbl": table_name},
    ).fetchall()
    for (conname,) in rows:
        conn.execute(text(f'ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS "{conname}"'))


def _migrate_integer_ids_to_uuid() -> None:
    """One-time migration: convert integer PK/FK columns to random UUIDs in place.

    Uses a temp mapping table + ALTER COLUMN TYPE so existing primary keys stay
    valid without creating new constraints (PostgreSQL 15+ public schema limits).
    """
    if not _column_is_integer_pk("users"):
        if _column_is_uuid_pk("users"):
            _cleanup_partial_uuid_migration_on_connect()
        return

    logger.warning("Migrating integer primary keys to random UUIDs — this may take a moment")
    child_tables = ("documents", "pending_documents", "user_text_entries")

    with engine.begin() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
        _cleanup_partial_uuid_migration(conn)

        conn.execute(
            text(
                """
                CREATE TEMP TABLE _int_user_map AS
                SELECT id AS old_id, gen_random_uuid() AS new_id FROM users
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION pg_temp.map_user_int(i integer)
                RETURNS uuid LANGUAGE sql STABLE AS $$
                    SELECT new_id FROM _int_user_map WHERE old_id = i
                $$
                """
            )
        )

        for child in child_tables:
            _drop_table_fk_constraints(conn, child)

        conn.execute(text("ALTER TABLE users ALTER COLUMN id DROP DEFAULT"))
        conn.execute(
            text("ALTER TABLE users ALTER COLUMN id TYPE uuid USING (pg_temp.map_user_int(id))")
        )

        for child in child_tables:
            conn.execute(
                text(
                    f"ALTER TABLE {child} ALTER COLUMN user_id TYPE uuid "
                    f"USING (pg_temp.map_user_int(user_id))"
                )
            )
            conn.execute(text(f"ALTER TABLE {child} ALTER COLUMN id DROP DEFAULT"))
            conn.execute(
                text(f"ALTER TABLE {child} ALTER COLUMN id TYPE uuid USING (gen_random_uuid())")
            )
            conn.execute(
                text(
                    f"ALTER TABLE {child} ADD CONSTRAINT {child}_user_id_fkey "
                    f"FOREIGN KEY (user_id) REFERENCES users(id)"
                )
            )
            conn.execute(text(f"DROP SEQUENCE IF EXISTS {child}_id_seq CASCADE"))

        if inspect(engine).has_table("prompts") and _column_is_integer_pk("prompts"):
            conn.execute(text("ALTER TABLE prompts ALTER COLUMN id DROP DEFAULT"))
            conn.execute(
                text("ALTER TABLE prompts ALTER COLUMN id TYPE uuid USING (gen_random_uuid())")
            )
            conn.execute(text("DROP SEQUENCE IF EXISTS prompts_id_seq CASCADE"))

        conn.execute(text("DROP SEQUENCE IF EXISTS users_id_seq CASCADE"))

    logger.info("Integer-to-UUID migration completed — reindex ChromaDB if you use vector search")


def _cleanup_partial_uuid_migration_on_connect() -> None:
    """Best-effort cleanup for leftover columns from a failed legacy migration."""
    try:
        with engine.begin() as conn:
            _cleanup_partial_uuid_migration(conn)
    except Exception as e:
        logger.debug("Partial UUID migration cleanup skipped: %s", e)


class User(Base):
    """User table - stores Telegram user details."""
    __tablename__ = "users"

    id = _uuid_pk()
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    username = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to documents
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, username={self.username})>"

    def to_dict(self):
        return {
            "id": as_str(self.id),
            "telegram_id": self.telegram_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Document(Base):
    """Document table - stores OCR extraction results."""
    __tablename__ = "documents"

    id = _uuid_pk()
    user_id = _uuid_fk("users.id")
    
    # Document metadata
    file_name = Column(String(500), nullable=True)
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)
    source = Column(String(20), nullable=False, default='telegram', index=True)
    telegram_file_unique_id = Column(String(255), nullable=True, index=True)
    content_sha256 = Column(String(64), nullable=True, index=True)
    dhash = Column(String(64), nullable=True, index=True)
    phash = Column(String(64), nullable=True, index=True)
    
    # OCR extracted data (stored as JSON string)
    extracted_data = Column(Text, nullable=True)
    document_type = Column(String(100), nullable=True)  # invoice, receipt, etc.
    
    # Key extracted fields (denormalized for easy querying)
    title = Column(String(500), nullable=True)
    document_date = Column(String(50), nullable=True)
    total_amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    vendor_name = Column(String(255), nullable=True)
    invoice_number = Column(String(100), nullable=True)
    gstin = Column(String(50), nullable=True)
    gst_amount = Column(Float, nullable=True)
    igst_amount = Column(Float, nullable=True)
    cgst_amount = Column(Float, nullable=True)
    sgst_amount = Column(Float, nullable=True)
    expense_category = Column(String(50), nullable=True)
    user_input_text = Column(Text, nullable=True)
    
    # Confidence score
    confidence_overall = Column(Float, nullable=True)
    
    # Raw text content
    raw_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to user
    user = relationship("User", back_populates="documents")

    def __repr__(self):
        return f"<Document(id={self.id}, user_id={self.user_id}, type={self.document_type})>"

    def to_dict(self):
        return {
            "id": as_str(self.id),
            "user_id": as_str(self.user_id),
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "source": self.source,
            "extracted_data": json.loads(self.extracted_data) if self.extracted_data else None,
            "document_type": self.document_type,
            "title": self.title,
            "document_date": self.document_date,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "vendor_name": self.vendor_name,
            "invoice_number": self.invoice_number,
            "gstin": self.gstin,
            "gst_amount": self.gst_amount,
            "igst_amount": self.igst_amount,
            "cgst_amount": self.cgst_amount,
            "sgst_amount": self.sgst_amount,
            "expense_category": self.expense_category,
            "user_input_text": self.user_input_text,
            "confidence_overall": self.confidence_overall,
            "raw_text": self.raw_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PendingDocument(Base):
    """Pending document table - stores documents awaiting user confirmation."""
    __tablename__ = "pending_documents"

    id = _uuid_pk()
    user_id = _uuid_fk("users.id")
    
    # Unique token for accessing this pending document (shared across Telegram/Web)
    token = Column(String(100), unique=True, nullable=False, index=True)
    
    # Source: 'telegram', 'whatsapp', or 'web'
    source = Column(String(20), nullable=False, default='web')
    
    # Document metadata
    file_name = Column(String(500), nullable=True)
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)
    telegram_file_unique_id = Column(String(255), nullable=True, index=True)
    content_sha256 = Column(String(64), nullable=True, index=True)
    dhash = Column(String(64), nullable=True, index=True)
    phash = Column(String(64), nullable=True, index=True)
    
    # OCR extracted data (stored as JSON string)
    extracted_data = Column(Text, nullable=True)
    user_input_text = Column(Text, nullable=True)
    expense_category = Column(String(50), nullable=True)
    gstin = Column(String(50), nullable=True)
    gst_amount = Column(Float, nullable=True)
    igst_amount = Column(Float, nullable=True)
    cgst_amount = Column(Float, nullable=True)
    sgst_amount = Column(Float, nullable=True)
    
    # Confidence score
    confidence_overall = Column(Float, nullable=True)
    
    # Telegram-specific fields (for callback handling)
    telegram_chat_id = Column(BigInteger, nullable=True)
    telegram_message_id = Column(Integer, nullable=True)
    telegram_file_id = Column(String(255), nullable=True)
    
    # OCR queue metadata
    ocr_job_id = Column(String(100), nullable=True, index=True)
    retry_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    ocr_started_at = Column(DateTime(timezone=True), nullable=True)
    ocr_completed_at = Column(DateTime(timezone=True), nullable=True)

    # Status: 'processing', 'ready', 'confirmed', 'cancelled', 'failed'
    status = Column(String(20), nullable=False, default='pending')
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Auto-expire after 24h

    # Relationship to user
    user = relationship("User")

    def __repr__(self):
        return f"<PendingDocument(id={self.id}, token={self.token}, user_id={self.user_id}, status={self.status})>"

    def to_dict(self):
        return {
            "id": as_str(self.id),
            "token": self.token,
            "user_id": as_str(self.user_id),
            "source": self.source,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "extracted_data": json.loads(self.extracted_data) if self.extracted_data else None,
            "user_input_text": self.user_input_text,
            "expense_category": self.expense_category,
            "gstin": self.gstin,
            "gst_amount": self.gst_amount,
            "igst_amount": self.igst_amount,
            "cgst_amount": self.cgst_amount,
            "sgst_amount": self.sgst_amount,
            "confidence_overall": self.confidence_overall,
            "telegram_chat_id": self.telegram_chat_id,
            "telegram_message_id": self.telegram_message_id,
            "telegram_file_id": self.telegram_file_id,
            "ocr_job_id": self.ocr_job_id,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "ocr_started_at": self.ocr_started_at.isoformat() if self.ocr_started_at else None,
            "ocr_completed_at": self.ocr_completed_at.isoformat() if self.ocr_completed_at else None,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class UserTextEntry(Base):
    """User plain-text entries, especially expense-related intents."""
    __tablename__ = "user_text_entries"

    id = _uuid_pk()
    user_id = _uuid_fk("users.id")
    text = Column(Text, nullable=False)
    source = Column(String(20), nullable=False, default='telegram', index=True)
    intent_tag = Column(String(100), nullable=True)
    expense_category = Column(String(50), nullable=True)
    amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")

    def to_dict(self):
        return {
            "id": as_str(self.id),
            "user_id": as_str(self.user_id),
            "text": self.text,
            "source": self.source,
            "intent_tag": self.intent_tag,
            "expense_category": self.expense_category,
            "amount": self.amount,
            "currency": self.currency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExpenseCategory(Base):
    """Canonical expense categories for OCR, text entries, and UI filters."""

    __tablename__ = "expense_categories"

    id = _uuid_pk()
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    display_order = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        return {
            "id": as_str(self.id),
            "name": self.name,
            "slug": self.slug,
            "display_order": self.display_order,
            "is_active": self.is_active,
        }


class BotPrompt(Base):
    """Editable LLM prompts (e.g. Telegram /q pipeline)."""

    __tablename__ = "prompts"

    id = _uuid_pk()
    prompt_key = Column(String(120), unique=True, nullable=False, index=True)
    label = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False, default="telegram_q")
    system_prompt = Column(Text, nullable=False, default="")
    user_prompt_template = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def init_db():
    """Initialize database - create all tables."""
    tables_without_categories = [
        t for name, t in Base.metadata.tables.items() if name != "expense_categories"
    ]
    Base.metadata.create_all(bind=engine, tables=tables_without_categories)
    _migrate_integer_ids_to_uuid()
    _rerandomize_deterministic_uuids()
    _ensure_hash_columns()
    _ensure_pending_queue_columns()
    _ensure_expense_category_columns()
    _ensure_user_input_text_columns()
    _ensure_user_text_entry_amount_columns()
    _ensure_source_columns()
    _ensure_gst_tax_columns()
    _ensure_expense_categories_table()
    _normalize_existing_expense_categories()
    _seed_telegram_q_prompts()
    _migrate_prompts_from_sqlite_wording()
    _migrate_prompt_emoji_rules()
    logger.info("Database initialized successfully")


def _ensure_hash_columns():
    """Lightweight migration to add hash columns on existing tables."""
    with engine.connect() as conn:
        for table in ("documents", "pending_documents"):
            cols = _table_column_names(table)
            if "telegram_file_unique_id" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN telegram_file_unique_id VARCHAR(255)"))
            if "content_sha256" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN content_sha256 VARCHAR(64)"))
            if "dhash" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN dhash VARCHAR(64)"))
            if "phash" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN phash VARCHAR(64)"))
        conn.commit()


def _ensure_pending_queue_columns():
    """Add queue-tracking columns on pending_documents for existing DBs."""
    ts_type = _timestamp_column_type_sql()
    with engine.connect() as conn:
        cols = _table_column_names("pending_documents")
        if "telegram_file_id" not in cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN telegram_file_id VARCHAR(255)"))
        if "ocr_job_id" not in cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN ocr_job_id VARCHAR(100)"))
        if "retry_count" not in cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN retry_count INTEGER DEFAULT 0"))
        if "error_message" not in cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN error_message TEXT"))
        if "ocr_started_at" not in cols:
            conn.execute(text(f"ALTER TABLE pending_documents ADD COLUMN ocr_started_at {ts_type}"))
        if "ocr_completed_at" not in cols:
            conn.execute(text(f"ALTER TABLE pending_documents ADD COLUMN ocr_completed_at {ts_type}"))
        conn.commit()


def _ensure_expense_category_columns():
    """Add expense_category column on documents and pending_documents for existing DBs."""
    with engine.connect() as conn:
        doc_cols = _table_column_names("documents")
        if "expense_category" not in doc_cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN expense_category VARCHAR(50)"))
        pending_cols = _table_column_names("pending_documents")
        if "expense_category" not in pending_cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN expense_category VARCHAR(50)"))
        conn.commit()


def _ensure_user_input_text_columns():
    """Add user_input_text columns for persisted user-entered upload text."""
    with engine.connect() as conn:
        doc_cols = _table_column_names("documents")
        if "user_input_text" not in doc_cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN user_input_text TEXT"))
        pending_cols = _table_column_names("pending_documents")
        if "user_input_text" not in pending_cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN user_input_text TEXT"))
        conn.commit()


def _ensure_user_text_entry_amount_columns():
    """Add amount/currency columns on user_text_entries for existing DBs."""
    with engine.connect() as conn:
        insp = inspect(engine)
        if not insp.has_table("user_text_entries"):
            conn.commit()
            return
        cols = _table_column_names("user_text_entries")
        if "amount" not in cols:
            conn.execute(text("ALTER TABLE user_text_entries ADD COLUMN amount FLOAT"))
        if "currency" not in cols:
            conn.execute(text("ALTER TABLE user_text_entries ADD COLUMN currency VARCHAR(10)"))
        conn.commit()


def _ensure_source_columns():
    """Add bot/source channel columns for existing databases."""
    with engine.connect() as conn:
        doc_cols = _table_column_names("documents")
        if "source" not in doc_cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN source VARCHAR(20) DEFAULT 'telegram' NOT NULL"))
        pending_cols = _table_column_names("pending_documents")
        if "source" not in pending_cols:
            conn.execute(text("ALTER TABLE pending_documents ADD COLUMN source VARCHAR(20) DEFAULT 'web' NOT NULL"))
        text_cols = _table_column_names("user_text_entries")
        if text_cols and "source" not in text_cols:
            conn.execute(text("ALTER TABLE user_text_entries ADD COLUMN source VARCHAR(20) DEFAULT 'telegram' NOT NULL"))
        conn.commit()


def _ensure_gst_tax_columns():
    """Add GST/tax breakdown columns on documents and pending_documents."""
    tax_cols = {
        "gst_amount": "FLOAT",
        "igst_amount": "FLOAT",
        "cgst_amount": "FLOAT",
        "sgst_amount": "FLOAT",
    }
    with engine.connect() as conn:
        for table in ("documents", "pending_documents"):
            cols = _table_column_names(table)
            if table == "pending_documents" and "gstin" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN gstin VARCHAR(50)"))
            for col, sql_type in tax_cols.items():
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}"))
        conn.commit()


_DEFAULT_EXPENSE_CATEGORY_NAMES = [
    "Food and Dining",
    "Groceries",
    "Rent",
    "Utilities",
    "Fual",
    "Shopping",
    "Entertainment",
    "Healthcare",
    "Edication",
    "Personal care",
    "Subscription",
    "EMI/Loans",
    "Insurance",
    "Investment",
    "Travel",
    "Savings",
    "CAB/Taxi",
    "Misecellaneous",
    "Other",
]

_category_cache: Optional[Dict[str, Any]] = None


def _slugify_category(name: str) -> str:
    s = name.lower().strip().replace("/", " ")
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-") or "other"


def _builtin_category_cache() -> Dict[str, Any]:
    """In-memory category list when expense_categories table is unavailable."""
    names = list(_DEFAULT_EXPENSE_CATEGORY_NAMES)
    return {
        "rows": [],
        "names": names,
        "by_name": {n: n for n in names},
        "by_slug": {_slugify_category(n): n for n in names},
        "by_lower": {n.lower(): n for n in names},
        "slug_by_name": {n: _slugify_category(n) for n in names},
    }


def _ensure_expense_categories_table() -> None:
    """Create public.expense_categories table and seed defaults if empty."""
    insp = inspect(engine)
    if not insp.has_table("expense_categories"):
        try:
            ExpenseCategory.__table__.create(bind=engine, checkfirst=True)
        except Exception as e:
            logger.warning(
                "Could not create expense_categories table (%s). Using built-in category list.",
                e,
            )
            DatabaseService.invalidate_category_cache()
            return

    db = SessionLocal()
    try:
        count = db.query(ExpenseCategory).count()
        if count == 0:
            for order, name in enumerate(_DEFAULT_EXPENSE_CATEGORY_NAMES):
                db.add(
                    ExpenseCategory(
                        name=name,
                        slug=_slugify_category(name),
                        display_order=order,
                        is_active=True,
                    )
                )
            db.commit()
            logger.info("Seeded %s expense categories", len(_DEFAULT_EXPENSE_CATEGORY_NAMES))
        else:
            for row in db.query(ExpenseCategory).filter(
                or_(ExpenseCategory.slug.is_(None), ExpenseCategory.slug == "")
            ).all():
                row.slug = _slugify_category(row.name)
            db.commit()
    finally:
        db.close()
    DatabaseService.invalidate_category_cache()


def _normalize_existing_expense_categories() -> None:
    """Normalize legacy/invalid expense_category values to canonical list or Other."""
    valid = set(DatabaseService.list_expense_category_names())
    with engine.begin() as conn:
        docs = conn.execute(
            text("SELECT id, expense_category FROM documents WHERE expense_category IS NOT NULL")
        ).mappings().all()
        for row in docs:
            normalized = DatabaseService.normalize_expense_category_label(row["expense_category"])
            if normalized != row["expense_category"] or normalized not in valid:
                conn.execute(
                    text("UPDATE documents SET expense_category = :c WHERE id = :id"),
                    {"c": normalized if normalized in valid else "Other", "id": row["id"]},
                )

        pending = conn.execute(
            text("SELECT id, expense_category FROM pending_documents WHERE expense_category IS NOT NULL")
        ).mappings().all()
        for row in pending:
            normalized = DatabaseService.normalize_expense_category_label(row["expense_category"])
            if normalized != row["expense_category"] or normalized not in valid:
                conn.execute(
                    text("UPDATE pending_documents SET expense_category = :c WHERE id = :id"),
                    {"c": normalized if normalized in valid else "Other", "id": row["id"]},
                )

        text_rows = conn.execute(
            text("SELECT id, expense_category FROM user_text_entries WHERE expense_category IS NOT NULL")
        ).mappings().all()
        for row in text_rows:
            normalized = DatabaseService.normalize_expense_category_label(row["expense_category"])
            if normalized != row["expense_category"] or normalized not in valid:
                conn.execute(
                    text("UPDATE user_text_entries SET expense_category = :c WHERE id = :id"),
                    {"c": normalized if normalized in valid else "Other", "id": row["id"]},
                )


def _migrate_prompts_from_sqlite_wording() -> None:
    """Replace DB prompt rows that still reference SQLite with PostgreSQL seed content."""
    try:
        from shared.telegram_q_prompt_seed import load_telegram_q_seed_rows
    except ImportError:
        return
    rows = load_telegram_q_seed_rows()
    if not rows:
        return
    try:
        updated = 0
        with engine.begin() as conn:
            for r in rows:
                pk = r.get("prompt_key")
                if not pk:
                    continue
                existing = conn.execute(
                    text("SELECT system_prompt FROM prompts WHERE prompt_key = :k"),
                    {"k": pk},
                ).scalar()
                if not existing or "sqlite" not in str(existing).lower():
                    continue
                conn.execute(
                    text(
                        """
                        UPDATE prompts
                        SET system_prompt = :sp,
                            user_prompt_template = :upt,
                            updated_at = NOW()
                        WHERE prompt_key = :pk
                        """
                    ),
                    {
                        "sp": r.get("system_prompt") or "",
                        "upt": r.get("user_prompt_template"),
                        "pk": pk,
                    },
                )
                updated += 1
        if updated:
            logger.info("Migrated %s prompt(s) from SQLite to PostgreSQL wording", updated)
            try:
                from shared.nlp_sql_service_v2 import invalidate_telegram_q_prompt_cache

                invalidate_telegram_q_prompt_cache()
            except Exception:
                pass
    except Exception as e:
        logger.warning("Prompt SQLite->PostgreSQL migration skipped: %s", e)


def _migrate_prompt_emoji_rules() -> None:
    """
    Overwrite the four response-formatting prompts with the latest emoji-rule version
    from the seed JSON. Runs every startup but only writes when content has changed,
    so it is safe to call repeatedly (idempotent on stable content).

    Prompts updated:
      - q_expense_save_confirm        (save confirmation — item-level emoji)
      - q_telegram_format_sql_response (SQL result formatter)
      - q_telegram_vector_semantic     (vector/semantic answer formatter)
      - q_telegram_social_reply        (greeting / small-talk)
    """
    EMOJI_RULE_KEYS = {
        "q_expense_save_confirm",
        "q_telegram_format_sql_response",
        "q_telegram_vector_semantic",
        "q_telegram_social_reply",
    }
    try:
        from shared.telegram_q_prompt_seed import load_telegram_q_seed_rows
    except ImportError:
        logger.warning("telegram_q_prompt_seed not available; skipping emoji-rule migration")
        return
    rows = load_telegram_q_seed_rows()
    if not rows:
        return
    seed_map = {r["prompt_key"]: r for r in rows if r.get("prompt_key") in EMOJI_RULE_KEYS}
    if not seed_map:
        return
    try:
        updated = 0
        with engine.begin() as conn:
            for pk, r in seed_map.items():
                new_sp = r.get("system_prompt") or ""
                new_upt = r.get("user_prompt_template")
                existing = conn.execute(
                    text("SELECT system_prompt FROM prompts WHERE prompt_key = :k"),
                    {"k": pk},
                ).scalar()
                if existing is None:
                    # Row not yet seeded — insert it
                    conn.execute(
                        text(
                            """
                            INSERT INTO prompts (prompt_key, label, category, system_prompt, user_prompt_template)
                            VALUES (:pk, :label, :cat, :sp, :upt)
                            """
                        ),
                        {
                            "pk": pk,
                            "label": r.get("label") or pk,
                            "cat": r.get("category") or "telegram_q",
                            "sp": new_sp,
                            "upt": new_upt,
                        },
                    )
                    updated += 1
                elif existing != new_sp:
                    conn.execute(
                        text(
                            """
                            UPDATE prompts
                            SET system_prompt = :sp,
                                user_prompt_template = :upt,
                                updated_at = NOW()
                            WHERE prompt_key = :pk
                            """
                        ),
                        {"sp": new_sp, "upt": new_upt, "pk": pk},
                    )
                    updated += 1
        if updated:
            logger.info("emoji-rule migration: updated %s prompt(s)", updated)
            try:
                from shared.nlp_sql_service_v2 import invalidate_telegram_q_prompt_cache
                invalidate_telegram_q_prompt_cache()
            except Exception:
                pass
        else:
            logger.debug("emoji-rule migration: all prompts already up-to-date")
    except Exception as e:
        logger.warning("emoji-rule prompt migration skipped: %s", e)


def _seed_telegram_q_prompts() -> None:
    """Insert default Telegram /q prompts if missing (idempotent)."""
    try:
        from shared.telegram_q_prompt_seed import load_telegram_q_seed_rows
    except ImportError:
        logger.warning("telegram_q_prompt_seed not available; skipping prompt seed")
        return
    rows = load_telegram_q_seed_rows()
    if not rows:
        logger.warning("No telegram_q seed rows; skipping prompt seed")
        return
    try:
        with engine.begin() as conn:
            for r in rows:
                pk = r.get("prompt_key")
                if not pk:
                    continue
                exists = conn.execute(
                    text("SELECT 1 FROM prompts WHERE prompt_key = :k"), {"k": pk}
                ).scalar()
                if exists:
                    continue
                conn.execute(
                    text(
                        """
                        INSERT INTO prompts (prompt_key, label, category, system_prompt, user_prompt_template)
                        VALUES (:prompt_key, :label, :category, :system_prompt, :user_prompt_template)
                        """
                    ),
                    {
                        "prompt_key": pk,
                        "label": r.get("label") or pk,
                        "category": r.get("category") or "telegram_q",
                        "system_prompt": r.get("system_prompt") or "",
                        "user_prompt_template": r.get("user_prompt_template"),
                    },
                )
        logger.info("Telegram /q prompts seed checked (%s definitions)", len(rows))
    except Exception as e:
        logger.warning("Prompt seed skipped or failed: %s", e)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


class DatabaseService:
    """Service class for database operations."""

    @staticmethod
    def invalidate_category_cache() -> None:
        global _category_cache
        _category_cache = None

    @staticmethod
    def _load_category_cache() -> Dict[str, Any]:
        global _category_cache
        if _category_cache is not None:
            return _category_cache

        if not inspect(engine).has_table("expense_categories"):
            _category_cache = _builtin_category_cache()
            return _category_cache

        db = get_db()
        try:
            rows = (
                db.query(ExpenseCategory)
                .filter(ExpenseCategory.is_active.is_(True))
                .order_by(ExpenseCategory.display_order, ExpenseCategory.name)
                .all()
            )
            if not rows:
                _category_cache = _builtin_category_cache()
                return _category_cache

            names = [r.name for r in rows]
            by_name = {r.name: r.name for r in rows}
            by_slug = {r.slug.lower(): r.name for r in rows}
            by_lower = {r.name.lower(): r.name for r in rows}
            slug_by_name = {r.name: r.slug for r in rows}
            _category_cache = {
                "rows": rows,
                "names": names,
                "by_name": by_name,
                "by_slug": by_slug,
                "by_lower": by_lower,
                "slug_by_name": slug_by_name,
            }
            return _category_cache
        except Exception as e:
            logger.warning(
                "Could not load expense_categories from DB (%s); using built-in list.",
                e,
            )
            _category_cache = _builtin_category_cache()
            return _category_cache
        finally:
            db.close()

    @staticmethod
    def list_expense_category_names() -> List[str]:
        return list(DatabaseService._load_category_cache()["names"])

    @staticmethod
    def list_expense_categories() -> List[Dict[str, Any]]:
        """Return active categories with id, name, slug (for API / UI)."""
        cache = DatabaseService._load_category_cache()
        if cache["rows"]:
            return [r.to_dict() for r in cache["rows"]]
        return [
            {
                "id": None,
                "name": name,
                "slug": _slugify_category(name),
                "display_order": idx,
                "is_active": True,
            }
            for idx, name in enumerate(cache["names"])
        ]

    @staticmethod
    def get_expense_categories_for_ai() -> List[str]:
        """Category names sent to OCR / NLP models."""
        return DatabaseService.list_expense_category_names()

    @staticmethod
    def get_category_slug(name: str) -> str:
        cache = DatabaseService._load_category_cache()
        return cache["slug_by_name"].get(name, _slugify_category(name))

    @staticmethod
    def normalize_expense_category_label(raw: Optional[str]) -> str:
        """Map name, slug, or alias to canonical category name from DB."""
        c = (raw or "").strip()
        if not c:
            return "Other"
        cache = DatabaseService._load_category_cache()
        if c in cache["by_name"]:
            return c
        slug_key = c.lower().replace("_", "-")
        if slug_key in cache["by_slug"]:
            return cache["by_slug"][slug_key]
        by_lower = cache["by_lower"]
        return by_lower.get(c.lower(), "Other")

    @staticmethod
    def resolve_category_filter(raw: Optional[str]) -> Optional[str]:
        """Resolve API ?category= (name or slug) to canonical name."""
        if not raw or not str(raw).strip():
            return None
        return DatabaseService.normalize_expense_category_label(raw)

    @staticmethod
    def _resolve_document_expense_category(
        data: dict,
        _merged_input_text: Optional[str],
        _file_name: Optional[str],
    ) -> str:
        """Category from OCR JSON `expense_category` (normalized)."""
        ocr_raw = data.get("expense_category")
        if isinstance(ocr_raw, str):
            return DatabaseService.normalize_expense_category_label(ocr_raw.strip())
        return DatabaseService.normalize_expense_category_label(None)

    @staticmethod
    def _infer_document_expense_category(
        data: dict,
        merged_input_text: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        """Resolve category from OCR JSON; LLM fallback when still Other."""
        category = DatabaseService._resolve_document_expense_category(
            data, merged_input_text, file_name
        )
        if category != "Other":
            return category

        amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
        vendor = data.get("vendor_or_sender") if isinstance(data.get("vendor_or_sender"), dict) else {}
        hint_parts = [
            merged_input_text,
            data.get("title"),
            data.get("document_type"),
            vendor.get("name"),
            file_name,
        ]
        if amounts.get("total"):
            hint_parts.append(f"paid {amounts.get('total')} {amounts.get('currency') or 'INR'}")
        hint = " ".join(str(p) for p in hint_parts if p).strip()
        if not hint:
            return "Other"

        try:
            from shared.nlp_sql_service_v2 import get_nlp_sql_service_v2

            decision = get_nlp_sql_service_v2().classify_plain_text_expense(hint)
            if decision.get("category"):
                return DatabaseService.normalize_expense_category_label(decision["category"])
        except Exception as e:
            logger.warning("LLM category fallback failed: %s", e)
        return "Other"

    @staticmethod
    def get_user_expense_items(
        user_id: uuid.UUID,
        category: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Unified expense rows from documents + manual text entries (for UI)."""
        db = get_db()
        try:
            items: List[Dict[str, Any]] = []
            docs = (
                db.query(Document)
                .filter(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
                .limit(limit)
                .all()
            )
            for doc in docs:
                cat = DatabaseService.normalize_expense_category_label(doc.expense_category)
                items.append({
                    "source": "document",
                    "channel": doc.source or "telegram",
                    "id": as_str(doc.id),
                    "expense_category": cat,
                    "payment": doc.total_amount,
                    "currency": doc.currency or "INR",
                    "title": doc.title or doc.file_name or "Document",
                    "vendor": doc.vendor_name,
                    "document_type": doc.document_type,
                    "date": doc.document_date or (
                        doc.created_at.isoformat() if doc.created_at else None
                    ),
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                })

            texts = (
                db.query(UserTextEntry)
                .filter(UserTextEntry.user_id == user_id)
                .order_by(UserTextEntry.created_at.desc())
                .limit(limit)
                .all()
            )
            for entry in texts:
                cat = DatabaseService.normalize_expense_category_label(entry.expense_category)
                items.append({
                    "source": "text",
                    "channel": entry.source or "telegram",
                    "id": as_str(entry.id),
                    "expense_category": cat,
                    "payment": entry.amount,
                    "currency": entry.currency or "INR",
                    "title": (entry.text or "")[:120],
                    "vendor": None,
                    "document_type": "text_entry",
                    "date": entry.created_at.isoformat() if entry.created_at else None,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                })

            if category:
                norm = DatabaseService.resolve_category_filter(category)
                items = [i for i in items if i["expense_category"] == norm]
            if source:
                source_norm = source.strip().lower()
                items = [i for i in items if (i.get("channel") or "").lower() == source_norm]

            items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            return items[:limit]
        finally:
            db.close()

    @staticmethod
    def update_document_expense_category(
        doc_id: uuid.UUID, user_id: uuid.UUID, expense_category: str
    ) -> Optional[Document]:
        """Update category on a saved document and refresh vector metadata."""
        db = get_db()
        try:
            doc = (
                db.query(Document)
                .filter(Document.id == doc_id, Document.user_id == user_id)
                .first()
            )
            if not doc:
                return None
            doc.expense_category = DatabaseService.normalize_expense_category_label(
                expense_category
            )
            if doc.extracted_data:
                try:
                    data = json.loads(doc.extracted_data)
                    if isinstance(data, dict):
                        data["expense_category"] = doc.expense_category
                        doc.extracted_data = json.dumps(data, ensure_ascii=False)
                except Exception:
                    pass
            db.commit()
            db.refresh(doc)
            DatabaseService.index_document_in_vector_store(doc)
            return doc
        except Exception as e:
            db.rollback()
            logger.error("update_document_expense_category failed: %s", e)
            raise
        finally:
            db.close()

    @staticmethod
    def _extract_amount_from_text(user_text: str) -> Optional[float]:
        """Extract a likely expense amount from free text (supports 8,000 / 8000.50)."""
        if not user_text:
            return None
        # Prefer larger numbers first; treat simple integers/decimals as INR by default.
        matches = re.findall(r"(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+(?:\.[0-9]{1,2})?)", user_text, flags=re.IGNORECASE)
        candidates: List[float] = []
        for m in matches:
            try:
                candidates.append(float(m.replace(",", "")))
            except Exception:
                continue
        if not candidates:
            return None
        # Heuristic: pick the largest monetary number in message.
        return max(candidates)

    @staticmethod
    def get_or_create_user(telegram_id: int, first_name: Optional[str], 
                           last_name: Optional[str], username: Optional[str]) -> User:
        """Get existing user or create new one."""
        db = get_db()
        try:
            # Try to find existing user
            user = db.query(User).filter(User.telegram_id == telegram_id).first()
            
            if user:
                # Update user info if changed
                if first_name and user.first_name != first_name:
                    user.first_name = first_name
                if last_name and user.last_name != last_name:
                    user.last_name = last_name
                if username and user.username != username:
                    user.username = username
                db.commit()
                db.refresh(user)
                logger.info(f"Updated user: {user}")
            else:
                # Create new user
                user = User(
                    telegram_id=telegram_id,
                    first_name=first_name,
                    last_name=last_name,
                    username=username
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Created new user: {user}")
            
            return user
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in get_or_create_user: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_user_by_id(user_id: uuid.UUID) -> Optional[User]:
        """Fetch user by internal users.id."""
        db = get_db()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    @staticmethod
    def _vendor_name_from_ocr_data(data: Any) -> Optional[str]:
        """Read vendor name from OCR JSON (nested or flat Gemini schema)."""
        if not isinstance(data, dict):
            return None
        vendor = data.get("vendor_or_sender")
        if isinstance(vendor, dict):
            name = DatabaseService._normalize_text(vendor.get("name"))
            if name:
                return name
        for key in ("vendor", "vendor_name"):
            flat_vendor = data.get(key)
            if isinstance(flat_vendor, str):
                name = DatabaseService._normalize_text(flat_vendor)
                if name:
                    return name
        return None

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def extract_tax_fields_from_ocr(data: Any) -> Dict[str, Any]:
        """Parse GST number and tax breakdown from OCR JSON."""
        if not isinstance(data, dict):
            return {
                "gstin": None,
                "gst_amount": None,
                "igst_amount": None,
                "cgst_amount": None,
                "sgst_amount": None,
            }

        amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
        identifiers = data.get("identifiers") if isinstance(data.get("identifiers"), dict) else {}
        taxes = data.get("taxes") if isinstance(data.get("taxes"), dict) else {}
        vendor = data.get("vendor_or_sender") if isinstance(data.get("vendor_or_sender"), dict) else {}

        gstin_raw = (
            identifiers.get("gstin")
            or identifiers.get("gst_number")
            or identifiers.get("gst_no")
            or data.get("gstin")
            or data.get("gst_number")
            or vendor.get("gstin")
        )
        gstin = str(gstin_raw).strip() if gstin_raw else None

        igst = DatabaseService._float_or_none(
            taxes.get("igst") or taxes.get("IGST") or amounts.get("igst") or amounts.get("IGST")
        )
        cgst = DatabaseService._float_or_none(
            taxes.get("cgst") or taxes.get("CGST") or amounts.get("cgst") or amounts.get("CGST")
        )
        sgst = DatabaseService._float_or_none(
            taxes.get("sgst") or taxes.get("SGST") or amounts.get("sgst") or amounts.get("SGST")
        )
        gst_amount = DatabaseService._float_or_none(
            amounts.get("gst")
            or amounts.get("gst_amount")
            or amounts.get("tax")
            or amounts.get("total_tax")
            or taxes.get("total")
            or taxes.get("gst")
            or taxes.get("total_gst")
        )

        if gst_amount is None and cgst is not None and sgst is not None:
            gst_amount = round(cgst + sgst, 2)
        elif gst_amount is None and igst is not None:
            gst_amount = igst

        return {
            "gstin": gstin,
            "gst_amount": gst_amount,
            "igst_amount": igst,
            "cgst_amount": cgst,
            "sgst_amount": sgst,
        }

    @staticmethod
    def _apply_tax_fields_to_model(model: Any, data: Any) -> None:
        tax = DatabaseService.extract_tax_fields_from_ocr(data)
        for key, value in tax.items():
            if hasattr(model, key):
                setattr(model, key, value)

    @staticmethod
    def backfill_document_vendor_names(user_id: uuid.UUID) -> int:
        """Populate documents.vendor_name from OCR JSON when the column is empty."""
        db = get_db()
        updated = 0
        try:
            docs = (
                db.query(Document)
                .filter(Document.user_id == user_id)
                .filter(or_(Document.vendor_name.is_(None), Document.vendor_name == ""))
                .filter(Document.extracted_data.isnot(None))
                .all()
            )
            for doc in docs:
                try:
                    parsed = json.loads(doc.extracted_data) if doc.extracted_data else {}
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                name = DatabaseService._vendor_name_from_ocr_data(parsed)
                if not name:
                    continue
                doc.vendor_name = name
                updated += 1
            if updated:
                db.commit()
                logger.info("Backfilled vendor_name for %s document(s), user_id=%s", updated, user_id)
            return updated
        except Exception as e:
            db.rollback()
            logger.error("backfill_document_vendor_names failed: %s", e)
            raise
        finally:
            db.close()

    @staticmethod
    def save_document(user_id: uuid.UUID, file_name: Optional[str], mime_type: Optional[str],
                      file_size: Optional[int], extracted_json: str,
                      raw_text: Optional[str] = None,
                      user_input_text: Optional[str] = None,
                      source: str = "telegram",
                      telegram_file_unique_id: Optional[str] = None,
                      content_sha256: Optional[str] = None,
                      dhash: Optional[str] = None,
                      phash: Optional[str] = None) -> Document:
        """Save document with OCR extraction results."""
        db = get_db()
        try:
            # Parse JSON to extract key fields
            try:
                data = json.loads(extracted_json)
            except:
                data = {}

            # Extract key fields for denormalized columns
            amounts = data.get("amounts", {})
            vendor = data.get("vendor_or_sender", {}) if isinstance(data.get("vendor_or_sender"), dict) else {}
            identifiers = data.get("identifiers", {}) if isinstance(data.get("identifiers"), dict) else {}
            confidence = data.get("confidence", {}) if isinstance(data.get("confidence"), dict) else {}
            vendor_name = DatabaseService._vendor_name_from_ocr_data(data)
            if not vendor_name and isinstance(vendor, dict):
                vendor_name = DatabaseService._normalize_text(vendor.get("name"))
            total_amount = amounts.get("total")
            if total_amount is None and data.get("total_amount") is not None:
                total_amount = data.get("total_amount")
            currency = amounts.get("currency") or data.get("currency")
            invoice_number = identifiers.get("invoice_number") or data.get("bill_number")
            tax = DatabaseService.extract_tax_fields_from_ocr(data)
            gstin = tax["gstin"]
            merged_input_text = " ".join([x for x in [raw_text, user_input_text] if x]).strip() or None
            classified_category = DatabaseService._infer_document_expense_category(
                data, merged_input_text, file_name
            )
            data["expense_category"] = classified_category
            stored_json = json.dumps(data, ensure_ascii=False) if data else extracted_json

            doc = Document(
                user_id=user_id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                source=(source or "telegram").strip().lower(),
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                extracted_data=stored_json,
                document_type=data.get("document_type"),
                title=data.get("title"),
                document_date=data.get("date"),
                total_amount=total_amount,
                currency=currency,
                vendor_name=vendor_name,
                invoice_number=invoice_number,
                gstin=gstin,
                gst_amount=tax["gst_amount"],
                igst_amount=tax["igst_amount"],
                cgst_amount=tax["cgst_amount"],
                sgst_amount=tax["sgst_amount"],
                expense_category=classified_category,
                user_input_text=user_input_text,
                confidence_overall=confidence.get("overall"),
                raw_text=raw_text or data.get("text_content")
            )
            
            db.add(doc)
            db.commit()
            db.refresh(doc)
            logger.info(f"Saved document: {doc}")

            # Chroma / OpenAI vector index (required for /q semantic search)
            DatabaseService.index_document_in_vector_store(doc)

            return doc
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in save_document: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def _vector_payload_from_document(doc: Document) -> Dict[str, Any]:
        """Shape expected by vector_service._document_to_text (extracted_json key)."""
        user = DatabaseService.get_user_by_id(doc.user_id)
        username = (getattr(user, "username", None) or "").strip() if user else ""
        return {
            "id": as_str(doc.id),
            "user_id": doc.user_id,
            "username": username,
            "file_name": doc.file_name,
            "source": doc.source,
            "document_type": doc.document_type,
            "title": doc.title,
            "vendor_name": doc.vendor_name,
            "expense_category": doc.expense_category,
            "user_input_text": doc.user_input_text,
            "total_amount": doc.total_amount,
            "extracted_json": doc.extracted_data,
            "document_date": doc.document_date,
            "raw_text": doc.raw_text,
        }

    @staticmethod
    def index_document_in_vector_store(doc: Document, attempts: int = 3) -> bool:
        """Upsert document embedding in Chroma after SQL commit. Retries on failure."""
        vs = get_vector_service()
        payload = DatabaseService._vector_payload_from_document(doc)
        for i in range(attempts):
            if vs.add_document(doc.id, doc.user_id, payload):
                return True
            logger.warning(
                "Vector upsert attempt %s/%s failed for document id=%s",
                i + 1,
                attempts,
                doc.id,
            )
            time.sleep(0.5 * (i + 1))
        logger.error(
            "VECTOR INDEX FAILED: document id=%s user_id=%s saved in SQL but semantic search will NOT find it "
            "(check OPENAI_API_KEY, network, CHROMA_DB_PATH). Re-run DatabaseService.reindex_document_vector(%s, %s).",
            doc.id,
            doc.user_id,
            doc.id,
            doc.user_id,
        )
        return False

    @staticmethod
    def index_user_text_entry_in_vector(entry: UserTextEntry, attempts: int = 3) -> bool:
        vs = get_vector_service()
        user = DatabaseService.get_user_by_id(entry.user_id)
        username = (getattr(user, "username", None) or "").strip() if user else ""
        for i in range(attempts):
            if vs.add_user_text_entry(
                entry_id=entry.id,
                user_id=entry.user_id,
                text=entry.text,
                intent_tag=entry.intent_tag or "expense_text",
                expense_category=entry.expense_category,
                username=username,
            ):
                return True
            logger.warning(
                "Vector upsert attempt %s/%s failed for user_text_entries id=%s",
                i + 1,
                attempts,
                entry.id,
            )
            time.sleep(0.5 * (i + 1))
        logger.error(
            "VECTOR INDEX FAILED: user_text_entries id=%s user_id=%s not in Chroma — manual expense text search broken.",
            entry.id,
            entry.user_id,
        )
        return False

    @staticmethod
    def _vector_payload_from_pending(pending: PendingDocument) -> Dict[str, Any]:
        """Shape for vector_service.add_pending_document."""
        user = DatabaseService.get_user_by_id(pending.user_id)
        username = (getattr(user, "username", None) or "").strip() if user else ""
        data: Dict[str, Any] = {}
        if pending.extracted_data:
            try:
                parsed = json.loads(pending.extracted_data)
                if isinstance(parsed, dict):
                    data = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
        return {
            "user_id": pending.user_id,
            "username": username,
            "file_name": pending.file_name,
            "source": pending.source,
            "title": pending.file_name,
            "vendor_name": DatabaseService._vendor_name_from_ocr_data(data),
            "expense_category": pending.expense_category or data.get("expense_category"),
            "user_input_text": pending.user_input_text,
            "total_amount": amounts.get("total"),
            "extracted_json": pending.extracted_data,
            "status": "pending",
        }

    @staticmethod
    def reindex_user_vectors(
        user_id: uuid.UUID,
        *,
        include_pending: bool = True,
        purge_first: bool = True,
    ) -> Dict[str, Any]:
        """Rebuild Chroma embeddings for one user from PostgreSQL (per-user guardrails)."""
        user = DatabaseService.get_user_by_id(user_id)
        if not user:
            return {"error": "user not found", "user_id": as_str(user_id)}

        vs = get_vector_service()
        stats: Dict[str, Any] = {
            "user_id": as_str(user_id),
            "username": user.username,
            "telegram_id": user.telegram_id,
            "purged": 0,
            "documents_ok": 0,
            "documents_fail": 0,
            "text_entries_ok": 0,
            "text_entries_fail": 0,
            "pending_ok": 0,
            "pending_fail": 0,
        }

        if purge_first:
            stats["purged"] = vs.purge_user_vectors(user_id)

        db = get_db()
        try:
            docs = db.query(Document).filter(Document.user_id == user_id).order_by(Document.created_at).all()
            for doc in docs:
                if DatabaseService.index_document_in_vector_store(doc):
                    stats["documents_ok"] += 1
                else:
                    stats["documents_fail"] += 1

            texts = (
                db.query(UserTextEntry)
                .filter(UserTextEntry.user_id == user_id)
                .order_by(UserTextEntry.created_at)
                .all()
            )
            for entry in texts:
                if DatabaseService.index_user_text_entry_in_vector(entry):
                    stats["text_entries_ok"] += 1
                else:
                    stats["text_entries_fail"] += 1

            if include_pending:
                pendings = (
                    db.query(PendingDocument)
                    .filter(
                        PendingDocument.user_id == user_id,
                        PendingDocument.status.in_(["pending", "processing", "ready"]),
                    )
                    .order_by(PendingDocument.created_at)
                    .all()
                )
                for pending in pendings:
                    payload = DatabaseService._vector_payload_from_pending(pending)
                    if vs.add_pending_document(pending.id, pending.user_id, payload):
                        stats["pending_ok"] += 1
                    else:
                        stats["pending_fail"] += 1
        finally:
            db.close()

        stats["total_indexed"] = (
            stats["documents_ok"] + stats["text_entries_ok"] + stats["pending_ok"]
        )
        stats["total_failed"] = (
            stats["documents_fail"] + stats["text_entries_fail"] + stats["pending_fail"]
        )
        return stats

    @staticmethod
    def reindex_all_users_vectors(
        *,
        include_pending: bool = True,
        purge_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """Rebuild Chroma embeddings for every user."""
        db = get_db()
        try:
            users = db.query(User).order_by(User.created_at).all()
        finally:
            db.close()

        results: List[Dict[str, Any]] = []
        for user in users:
            results.append(
                DatabaseService.reindex_user_vectors(
                    user.id,
                    include_pending=include_pending,
                    purge_first=purge_first,
                )
            )
        return results

    @staticmethod
    def count_user_vector_sources(user_id: uuid.UUID) -> Dict[str, int]:
        """Count SQL rows that would be embedded for a user (dry-run helper)."""
        db = get_db()
        try:
            doc_count = db.query(Document).filter(Document.user_id == user_id).count()
            text_count = db.query(UserTextEntry).filter(UserTextEntry.user_id == user_id).count()
            pending_count = (
                db.query(PendingDocument)
                .filter(
                    PendingDocument.user_id == user_id,
                    PendingDocument.status.in_(["pending", "processing", "ready"]),
                )
                .count()
            )
            return {
                "documents": doc_count,
                "text_entries": text_count,
                "pending": pending_count,
            }
        finally:
            db.close()

    @staticmethod
    def reindex_document_vector(doc_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Reload document from SQL and push to Chroma (recovery after failed indexing)."""
        doc = DatabaseService.get_document_by_id(doc_id, user_id)
        if not doc:
            logger.warning("reindex_document_vector: no document id=%s for user_id=%s", doc_id, user_id)
            return False
        return DatabaseService.index_document_in_vector_store(doc)

    @staticmethod
    def save_user_text_entry(
        user_id: uuid.UUID,
        user_text: str,
        intent_tag: str = "expense_text",
        expense_category: Optional[str] = None,
        source: str = "telegram",
    ) -> Optional[UserTextEntry]:
        """Persist expense-related user text and index it in vector DB."""
        db = get_db()
        try:
            cleaned = (user_text or "").strip()
            if not cleaned:
                return None
            cat_candidate = (expense_category or "").strip()
            if cat_candidate:
                category = DatabaseService.normalize_expense_category_label(cat_candidate)
            else:
                category = DatabaseService._infer_document_expense_category(
                    {"title": cleaned}, merged_input_text=cleaned, file_name=None
                )
            amount = DatabaseService._extract_amount_from_text(cleaned)
            duplicate_after = datetime.now(timezone.utc) - timedelta(minutes=15)
            duplicate = (
                db.query(UserTextEntry)
                .filter(UserTextEntry.user_id == user_id)
                .filter(func.lower(UserTextEntry.text) == cleaned.lower())
                .filter(UserTextEntry.expense_category == category)
                .filter(UserTextEntry.amount == amount)
                .filter(UserTextEntry.created_at >= duplicate_after)
                .order_by(UserTextEntry.created_at.desc())
                .first()
            )
            if duplicate:
                logger.info(
                    "Skipped duplicate user text entry: id=%s, user_id=%s, category=%s",
                    duplicate.id,
                    user_id,
                    category,
                )
                return duplicate

            entry = UserTextEntry(
                user_id=user_id,
                text=cleaned,
                source=(source or "telegram").strip().lower(),
                intent_tag=intent_tag,
                expense_category=category,
                amount=amount,
                currency="INR" if amount is not None else None
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)

            DatabaseService.index_user_text_entry_in_vector(entry)

            logger.info(f"Saved user text entry: id={entry.id}, user_id={user_id}, category={category}")
            return entry
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in save_user_text_entry: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_user_documents(user_id: uuid.UUID, limit: int = 50):
        """Get all documents for a user."""
        db = get_db()
        try:
            docs = db.query(Document).filter(Document.user_id == user_id)\
                     .order_by(Document.created_at.desc()).limit(limit).all()
            return docs
        finally:
            db.close()

    @staticmethod
    def get_document_by_id(doc_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Document]:
        """Get specific document by ID (with user verification)."""
        db = get_db()
        try:
            return db.query(Document).filter(
                Document.id == doc_id,
                Document.user_id == user_id
            ).first()
        finally:
            db.close()

    @staticmethod
    def fetch_documents_for_vector_enrichment(user_id: uuid.UUID, doc_ids: List[uuid.UUID]) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Batch-load documents for semantic / vector search enrichment."""
        if not doc_ids:
            return {}
        db = get_db()
        try:
            docs = (
                db.query(Document)
                .filter(Document.user_id == user_id, Document.id.in_(doc_ids))
                .all()
            )
            out: Dict[uuid.UUID, Dict[str, Any]] = {}
            for d in docs:
                out[d.id] = {
                    "id": as_str(d.id),
                "source": d.source,
                    "document_type": d.document_type,
                    "title": d.title,
                    "total_amount": d.total_amount,
                    "vendor_name": d.vendor_name,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "document_date": d.document_date,
                    "extracted_data": d.extracted_data,
                }
            return out
        finally:
            db.close()

    @staticmethod
    def fetch_user_text_entries_for_vector_enrichment(
        user_id: uuid.UUID, entry_ids: List[uuid.UUID]
    ) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Batch-load manual text expense rows for vector enrichment."""
        if not entry_ids:
            return {}
        db = get_db()
        try:
            rows = (
                db.query(UserTextEntry)
                .filter(UserTextEntry.user_id == user_id, UserTextEntry.id.in_(entry_ids))
                .all()
            )
            out: Dict[uuid.UUID, Dict[str, Any]] = {}
            for t in rows:
                out[t.id] = {
                    "id": as_str(t.id),
                    "text": t.text,
                "source": t.source,
                    "amount": t.amount,
                    "currency": t.currency,
                    "expense_category": t.expense_category,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
            return out
        finally:
            db.close()

    @staticmethod
    def get_distinct_vendor_names(user_id: uuid.UUID, limit: int = 500) -> List[str]:
        """Distinct vendor names for a user (vendor_name column + OCR JSON fallback)."""
        DatabaseService.backfill_document_vendor_names(user_id)
        db = get_db()
        try:
            docs = (
                db.query(Document)
                .filter(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
                .limit(2000)
                .all()
            )
            names: dict[str, str] = {}
            for doc in docs:
                label = (doc.vendor_name or "").strip()
                if not label and doc.extracted_data:
                    try:
                        parsed = json.loads(doc.extracted_data)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        label = DatabaseService._vendor_name_from_ocr_data(parsed) or ""
                if not label and doc.title:
                    label = str(doc.title).strip()
                if label:
                    key = label.lower()
                    if key not in names:
                        names[key] = label
                if len(names) >= limit:
                    break
            return list(names.values())[:limit]
        finally:
            db.close()

    @staticmethod
    def count_unique_vendors_for_user(user_id: uuid.UUID) -> int:
        return len(DatabaseService.get_distinct_vendor_names(user_id))

    @staticmethod
    def get_user_summary_stats(user_id: uuid.UUID) -> dict:
        """Get summary statistics for user's documents."""
        db = get_db()
        try:
            # Total document count
            total_docs = db.query(Document).filter(
                Document.user_id == user_id
            ).count()

            # Total amount (sum of all document amounts)
            total_amount = db.query(func.sum(Document.total_amount)).filter(
                Document.user_id == user_id,
                Document.total_amount.isnot(None)
            ).scalar() or 0.0

            # Document types breakdown
            doc_types = db.query(
                Document.document_type,
                func.count(Document.id).label('count')
            ).filter(
                Document.user_id == user_id
            ).group_by(Document.document_type).all()

            # Vendor count (unique vendors)
            vendor_count = DatabaseService.count_unique_vendors_for_user(user_id)

            # Recent documents (last 5)
            recent_docs = db.query(Document).filter(
                Document.user_id == user_id
            ).order_by(Document.created_at.desc()).limit(5).all()

            return {
                "total_documents": total_docs,
                "total_amount": round(total_amount, 2),
                "unique_vendors": vendor_count,
                "document_types": [{"type": t[0] or "unknown", "count": t[1]} for t in doc_types],
                "recent_documents": [
                    {
                        "id": as_str(d.id),
                        "title": d.title or d.file_name or "Untitled",
                        "type": d.document_type or "document",
                        "amount": d.total_amount,
                        "created_at": d.created_at.isoformat() if d.created_at else None
                    }
                    for d in recent_docs
                ]
            }
        finally:
            db.close()

    @staticmethod
    def create_pending_document(user_id: uuid.UUID, file_name: Optional[str], mime_type: Optional[str],
                                file_size: Optional[int], extracted_json: str,
                                confidence_overall: Optional[float] = None,
                                user_input_text: Optional[str] = None,
                                source: str = 'web',
                                telegram_chat_id: Optional[int] = None,
                                telegram_message_id: Optional[int] = None,
                                telegram_file_id: Optional[str] = None,
                                telegram_file_unique_id: Optional[str] = None,
                                content_sha256: Optional[str] = None,
                                dhash: Optional[str] = None,
                                phash: Optional[str] = None,
                                status: str = 'pending',
                                ocr_job_id: Optional[str] = None) -> PendingDocument:
        """Create a pending document awaiting user confirmation."""
        db = get_db()
        try:
            # Generate unique token
            token = secrets.token_urlsafe(32)
            
            # Set expiration to 24 hours from now
            expires_at = datetime.utcnow() + timedelta(hours=24)

            try:
                parsed = json.loads(extracted_json) if extracted_json else {}
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            pending_category = DatabaseService._infer_document_expense_category(
                parsed,
                merged_input_text=user_input_text,
                file_name=file_name,
            )
            parsed["expense_category"] = pending_category
            extracted_json = json.dumps(parsed, ensure_ascii=False)

            now = datetime.utcnow()
            has_ocr_result = bool(
                extracted_json and extracted_json.strip() not in ("", "{}", "null")
            )
            effective_job_id = ocr_job_id
            ocr_started = None
            ocr_completed = None

            if status == "processing":
                ocr_started = now
            elif has_ocr_result and status in ("pending", "ready"):
                ocr_started = now
                ocr_completed = now
                if not effective_job_id:
                    effective_job_id = "sync"

            tax = DatabaseService.extract_tax_fields_from_ocr(parsed)
            
            pending = PendingDocument(
                user_id=user_id,
                token=token,
                source=source,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                extracted_data=extracted_json,
                user_input_text=user_input_text,
                expense_category=pending_category,
                gstin=tax["gstin"],
                gst_amount=tax["gst_amount"],
                igst_amount=tax["igst_amount"],
                cgst_amount=tax["cgst_amount"],
                sgst_amount=tax["sgst_amount"],
                confidence_overall=confidence_overall,
                telegram_chat_id=telegram_chat_id,
                telegram_message_id=telegram_message_id,
                telegram_file_id=telegram_file_id,
                status=status,
                ocr_job_id=effective_job_id,
                ocr_started_at=ocr_started,
                ocr_completed_at=ocr_completed,
                expires_at=expires_at
            )
            
            db.add(pending)
            db.commit()
            db.refresh(pending)
            logger.info(f"Created pending document: {pending}")
            return pending
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in create_pending_document: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_pending_document_by_token(token: str) -> Optional[PendingDocument]:
        """Get pending document by token."""
        db = get_db()
        try:
            return db.query(PendingDocument).filter(
                PendingDocument.token == token,
                PendingDocument.status.in_(['pending', 'ready'])
            ).first()
        finally:
            db.close()

    @staticmethod
    def get_pending_document_by_id(pending_id: uuid.UUID, user_id: uuid.UUID) -> Optional[PendingDocument]:
        """Get pending document by ID (with user verification)."""
        db = get_db()
        try:
            return db.query(PendingDocument).filter(
                PendingDocument.id == pending_id,
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(['pending', 'ready'])
            ).first()
        finally:
            db.close()

    @staticmethod
    def get_pending_document_for_job(pending_id: uuid.UUID) -> Optional[PendingDocument]:
        """Get pending document for background processing regardless of user/session."""
        db = get_db()
        try:
            return db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
        finally:
            db.close()

    @staticmethod
    def get_user_text_entries(user_id: uuid.UUID, limit: int = 50):
        """Get manual text expense entries for a user."""
        db = get_db()
        try:
            return (
                db.query(UserTextEntry)
                .filter(UserTextEntry.user_id == user_id)
                .order_by(UserTextEntry.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

    @staticmethod
    def get_user_pending_documents(user_id: uuid.UUID, limit: int = 20):
        """Get all pending documents for a user."""
        db = get_db()
        try:
            pendings = db.query(PendingDocument).filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(['pending', 'processing', 'ready'])
            ).order_by(PendingDocument.created_at.desc()).limit(limit).all()
            return pendings
        finally:
            db.close()

    @staticmethod
    def confirm_pending_document(pending_id: uuid.UUID) -> Optional[Document]:
        """Confirm a pending document and save it to the documents table."""
        db = get_db()
        try:
            # Get pending document
            pending = db.query(PendingDocument).filter(
                PendingDocument.id == pending_id,
                PendingDocument.status.in_(['pending', 'ready'])
            ).first()
            
            if not pending:
                logger.warning(f"Pending document {pending_id} not found or not pending")
                return None
            
            # Create actual document
            doc = DatabaseService.save_document(
                user_id=pending.user_id,
                file_name=pending.file_name,
                mime_type=pending.mime_type,
                file_size=pending.file_size,
                extracted_json=pending.extracted_data,
                user_input_text=pending.user_input_text,
                source=pending.source,
                telegram_file_unique_id=pending.telegram_file_unique_id,
                content_sha256=pending.content_sha256,
                dhash=pending.dhash,
                phash=pending.phash
            )
            
            # Mark pending as confirmed
            pending.status = 'confirmed'
            db.commit()
            
            logger.info(f"Confirmed pending document {pending_id} -> document {doc.id}")
            return doc
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in confirm_pending_document: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def update_pending_document(pending_id: uuid.UUID, updated_json: str) -> Optional[PendingDocument]:
        """Update the extracted data of a pending document."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(
                PendingDocument.id == pending_id,
                PendingDocument.status.in_(['pending', 'ready'])
            ).first()
            
            if not pending:
                logger.warning(f"Pending document {pending_id} not found or not pending")
                return None
            
            pending.extracted_data = updated_json
            try:
                parsed = json.loads(updated_json) if updated_json else {}
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                cat = DatabaseService._infer_document_expense_category(
                    parsed,
                    merged_input_text=pending.user_input_text,
                    file_name=pending.file_name,
                )
                parsed["expense_category"] = cat
                pending.expense_category = cat
                pending.extracted_data = json.dumps(parsed, ensure_ascii=False)
                DatabaseService._apply_tax_fields_to_model(pending, parsed)
            db.commit()
            db.refresh(pending)
            
            logger.info(f"Updated pending document {pending_id}")
            return pending
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in update_pending_document: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def count_user_inflight_pending_documents(user_id: uuid.UUID) -> int:
        """Count pending OCR jobs currently in processing state for a user."""
        db = get_db()
        try:
            return db.query(PendingDocument).filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status == 'processing'
            ).count()
        finally:
            db.close()

    @staticmethod
    def set_pending_job_id(pending_id: uuid.UUID, job_id: str) -> Optional[PendingDocument]:
        """Attach Celery job id to pending document."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
            if not pending:
                return None
            pending.ocr_job_id = job_id
            if not pending.ocr_started_at:
                pending.ocr_started_at = datetime.utcnow()
            db.commit()
            db.refresh(pending)
            return pending
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in set_pending_job_id: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def set_pending_telegram_message_id(pending_id: uuid.UUID, message_id: int) -> Optional[PendingDocument]:
        """Persist Telegram message id linked to the pending workflow."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
            if not pending:
                return None
            pending.telegram_message_id = message_id
            db.commit()
            db.refresh(pending)
            return pending
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in set_pending_telegram_message_id: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def mark_pending_ocr_ready(pending_id: uuid.UUID, extracted_json: str, confidence_overall: Optional[float]) -> Optional[PendingDocument]:
        """Mark pending OCR job complete and ready for user confirmation."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
            if not pending or pending.status in ('confirmed', 'cancelled'):
                return None
            parsed: Dict[str, Any]
            try:
                parsed = json.loads(extracted_json) if extracted_json else {}
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            category = DatabaseService._infer_document_expense_category(
                parsed,
                merged_input_text=pending.user_input_text,
                file_name=pending.file_name,
            )
            parsed["expense_category"] = category
            pending.extracted_data = json.dumps(parsed, ensure_ascii=False)
            pending.expense_category = category
            DatabaseService._apply_tax_fields_to_model(pending, parsed)
            pending.confidence_overall = confidence_overall
            pending.status = 'ready'
            pending.error_message = None
            if not pending.ocr_started_at:
                pending.ocr_started_at = datetime.utcnow()
            pending.ocr_completed_at = datetime.utcnow()
            db.commit()
            db.refresh(pending)
            return pending
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in mark_pending_ocr_ready: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def mark_pending_ocr_failed(pending_id: uuid.UUID, error_message: str, retry_count: int = 0) -> Optional[PendingDocument]:
        """Mark pending OCR job failed after retries."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
            if not pending or pending.status in ('confirmed', 'cancelled'):
                return None
            pending.status = 'failed'
            pending.retry_count = retry_count
            pending.error_message = error_message[:1000] if error_message else "OCR job failed"
            pending.ocr_completed_at = datetime.utcnow()
            db.commit()
            db.refresh(pending)
            return pending
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in mark_pending_ocr_failed: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def _hamming_distance(hex_a: str, hex_b: str) -> int:
        """Compute Hamming distance between two equal-length hex hashes."""
        try:
            a = int(hex_a, 16)
            b = int(hex_b, 16)
            return (a ^ b).bit_count()
        except Exception:
            return 999

    @staticmethod
    def _normalize_text(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return re.sub(r"\s+", " ", str(value).strip().lower())

    @staticmethod
    def _build_fingerprint_from_extracted_json(extracted_json: Optional[str]) -> dict:
        """Extract comparable fields from OCR JSON for duplicate matching."""
        if not extracted_json:
            return {}
        try:
            data = json.loads(extracted_json) if isinstance(extracted_json, str) else extracted_json
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        amounts = data.get("amounts") or {}
        vendor = data.get("vendor_or_sender") or {}
        if not isinstance(vendor, dict):
            vendor = {}
        identifiers = data.get("identifiers") or {}
        if not isinstance(identifiers, dict):
            identifiers = {}

        total_amount = amounts.get("total")
        if total_amount is None and data.get("total_amount") is not None:
            try:
                total_amount = float(data.get("total_amount"))
            except Exception:
                total_amount = None
        else:
            try:
                total_amount = float(total_amount) if total_amount is not None else None
            except Exception:
                total_amount = None

        return {
            "vendor_name": DatabaseService._vendor_name_from_ocr_data(data)
            or DatabaseService._normalize_text(vendor.get("name")),
            "invoice_number": DatabaseService._normalize_text(identifiers.get("invoice_number")),
            "date": DatabaseService._normalize_text(data.get("date")),
            "title": DatabaseService._normalize_text(data.get("title")),
            "total_amount": total_amount
        }

    @staticmethod
    def find_duplicate_image_for_user(user_id: uuid.UUID,
                                      telegram_file_unique_id: Optional[str] = None,
                                      content_sha256: Optional[str] = None,
                                      dhash: Optional[str] = None,
                                      phash: Optional[str] = None,
                                      max_dhash_distance: int = 8,
                                      max_phash_distance: int = 8) -> Optional[dict]:
        """Find likely duplicate image for the same user using dHash + pHash."""
        if not telegram_file_unique_id and not content_sha256 and (not dhash or not phash):
            return None
        db = get_db()
        try:
            # Telegram-native stable file identity check (best for repeated uploads).
            if telegram_file_unique_id:
                existing = db.query(Document).filter(
                    Document.user_id == user_id,
                    Document.telegram_file_unique_id == telegram_file_unique_id
                ).first()
                if existing:
                    return {
                        "source": "documents",
                        "id": existing.id,
                        "file_name": existing.file_name,
                        "created_at": existing.created_at.isoformat() if existing.created_at else None,
                        "match_type": "telegram_file_unique_id"
                    }
                existing_pending = db.query(PendingDocument).filter(
                    PendingDocument.user_id == user_id,
                    PendingDocument.status.in_(['pending', 'processing', 'ready']),
                    PendingDocument.telegram_file_unique_id == telegram_file_unique_id
                ).first()
                if existing_pending:
                    return {
                        "source": "pending_documents",
                        "id": existing_pending.id,
                        "file_name": existing_pending.file_name,
                        "created_at": existing_pending.created_at.isoformat() if existing_pending.created_at else None,
                        "match_type": "telegram_file_unique_id"
                    }

            # Exact byte-level duplicate check via SHA-256 (works even without PIL/ImageHash)
            if content_sha256:
                existing = db.query(Document).filter(
                    Document.user_id == user_id,
                    Document.content_sha256 == content_sha256
                ).first()
                if existing:
                    return {
                        "source": "documents",
                        "id": existing.id,
                        "file_name": existing.file_name,
                        "created_at": existing.created_at.isoformat() if existing.created_at else None,
                        "match_type": "sha256_exact"
                    }

                existing_pending = db.query(PendingDocument).filter(
                    PendingDocument.user_id == user_id,
                    PendingDocument.status.in_(['pending', 'processing', 'ready']),
                    PendingDocument.content_sha256 == content_sha256
                ).first()
                if existing_pending:
                    return {
                        "source": "pending_documents",
                        "id": existing_pending.id,
                        "file_name": existing_pending.file_name,
                        "created_at": existing_pending.created_at.isoformat() if existing_pending.created_at else None,
                        "match_type": "sha256_exact"
                    }

            # Perceptual duplicate check (if hashes available)
            if not dhash or not phash:
                return None

            # Check confirmed documents first
            docs = db.query(Document).filter(
                Document.user_id == user_id,
                Document.dhash.isnot(None),
                Document.phash.isnot(None)
            ).all()
            for d in docs:
                dd = DatabaseService._hamming_distance(dhash, d.dhash)
                pd = DatabaseService._hamming_distance(phash, d.phash)
                if dd <= max_dhash_distance and pd <= max_phash_distance:
                    return {
                        "source": "documents",
                        "id": as_str(d.id),
                        "file_name": d.file_name,
                        "created_at": d.created_at.isoformat() if d.created_at else None,
                        "dhash_distance": dd,
                        "phash_distance": pd
                    }

            # Check pending documents
            pending_docs = db.query(PendingDocument).filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(['pending', 'processing', 'ready']),
                PendingDocument.dhash.isnot(None),
                PendingDocument.phash.isnot(None)
            ).all()
            for p in pending_docs:
                dd = DatabaseService._hamming_distance(dhash, p.dhash)
                pd = DatabaseService._hamming_distance(phash, p.phash)
                if dd <= max_dhash_distance and pd <= max_phash_distance:
                    return {
                        "source": "pending_documents",
                        "id": p.id,
                        "file_name": p.file_name,
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                        "dhash_distance": dd,
                        "phash_distance": pd
                    }
            return None
        finally:
            db.close()

    @staticmethod
    def find_duplicate_by_extracted_fingerprint(user_id: uuid.UUID, extracted_json: str) -> Optional[dict]:
        """
        Fallback duplicate detection using OCR-extracted business fields.
        Useful when Telegram file identifiers/bytes differ across uploads.
        """
        fp = DatabaseService._build_fingerprint_from_extracted_json(extracted_json)
        if not fp:
            return None

        vendor = fp.get("vendor_name")
        invoice_number = fp.get("invoice_number")
        date = fp.get("date")
        title = fp.get("title")
        total_amount = fp.get("total_amount")

        db = get_db()
        try:
            docs = db.query(Document).filter(
                Document.user_id == user_id,
                Document.extracted_data.isnot(None)
            ).order_by(Document.created_at.desc()).limit(200).all()

            for d in docs:
                existing = DatabaseService._build_fingerprint_from_extracted_json(d.extracted_data)
                if not existing:
                    continue
                e_vendor = existing.get("vendor_name")
                e_invoice = existing.get("invoice_number")
                e_date = existing.get("date")
                e_title = existing.get("title")
                e_total = existing.get("total_amount")

                amount_match = (
                    total_amount is not None and e_total is not None and abs(total_amount - e_total) < 0.01
                )
                # Strong signals
                if invoice_number and e_invoice and invoice_number == e_invoice and (amount_match or (vendor and e_vendor and vendor == e_vendor)):
                    return {"source": "documents", "id": as_str(d.id), "file_name": d.file_name, "match_type": "ocr_fingerprint_invoice"}
                if vendor and e_vendor and vendor == e_vendor and amount_match and date and e_date and date == e_date:
                    return {"source": "documents", "id": as_str(d.id), "file_name": d.file_name, "match_type": "ocr_fingerprint_vendor_amount_date"}
                if title and e_title and title == e_title and amount_match and date and e_date and date == e_date:
                    return {"source": "documents", "id": as_str(d.id), "file_name": d.file_name, "match_type": "ocr_fingerprint_title_amount_date"}

            pending_docs = db.query(PendingDocument).filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status.in_(['pending', 'processing', 'ready']),
                PendingDocument.extracted_data.isnot(None)
            ).order_by(PendingDocument.created_at.desc()).limit(200).all()

            for p in pending_docs:
                existing = DatabaseService._build_fingerprint_from_extracted_json(p.extracted_data)
                if not existing:
                    continue
                e_vendor = existing.get("vendor_name")
                e_invoice = existing.get("invoice_number")
                e_date = existing.get("date")
                e_title = existing.get("title")
                e_total = existing.get("total_amount")
                amount_match = (
                    total_amount is not None and e_total is not None and abs(total_amount - e_total) < 0.01
                )
                if invoice_number and e_invoice and invoice_number == e_invoice and (amount_match or (vendor and e_vendor and vendor == e_vendor)):
                    return {"source": "pending_documents", "id": p.id, "file_name": p.file_name, "match_type": "ocr_fingerprint_invoice"}
                if vendor and e_vendor and vendor == e_vendor and amount_match and date and e_date and date == e_date:
                    return {"source": "pending_documents", "id": p.id, "file_name": p.file_name, "match_type": "ocr_fingerprint_vendor_amount_date"}
                if title and e_title and title == e_title and amount_match and date and e_date and date == e_date:
                    return {"source": "pending_documents", "id": p.id, "file_name": p.file_name, "match_type": "ocr_fingerprint_title_amount_date"}

            return None
        finally:
            db.close()

    @staticmethod
    def cancel_pending_document(pending_id: uuid.UUID) -> bool:
        """Cancel a pending document."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(
                PendingDocument.id == pending_id,
                PendingDocument.status.in_(['pending', 'processing', 'ready', 'failed'])
            ).first()
            
            if not pending:
                return False
            
            pending.status = 'cancelled'
            db.commit()
            
            logger.info(f"Cancelled pending document {pending_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in cancel_pending_document: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def cleanup_expired_pending_documents() -> int:
        """Remove expired pending documents (older than 24h)."""
        db = get_db()
        try:
            now = datetime.utcnow()
            expired = db.query(PendingDocument).filter(
                PendingDocument.status.in_(['pending', 'processing', 'ready', 'failed']),
                PendingDocument.expires_at < now
            ).all()
            
            count = len(expired)
            for pending in expired:
                pending.status = 'cancelled'
            
            db.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired pending documents")
            
            return count
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in cleanup_expired_pending_documents: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def list_prompts(category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return prompt rows for admin UI (optionally filter by category)."""
        with engine.connect() as conn:
            if category:
                rows = conn.execute(
                    text(
                        """
                        SELECT id, prompt_key, label, category, system_prompt, user_prompt_template,
                               created_at, updated_at
                        FROM prompts WHERE category = :c ORDER BY prompt_key
                        """
                    ),
                    {"c": category},
                ).mappings().all()
            else:
                rows = conn.execute(
                    text(
                        """
                        SELECT id, prompt_key, label, category, system_prompt, user_prompt_template,
                               created_at, updated_at
                        FROM prompts ORDER BY category, prompt_key
                        """
                    )
                ).mappings().all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("id") is not None:
                d["id"] = as_str(d["id"])
            for key in ("created_at", "updated_at"):
                v = d.get(key)
                if v is not None and hasattr(v, "isoformat"):
                    d[key] = v.isoformat()
            out.append(d)
        return out

    @staticmethod
    def update_prompt_by_key(
        prompt_key: str, system_prompt: str, user_prompt_template: Optional[str] = None
    ) -> bool:
        """Persist prompt edits and invalidate in-process NLP prompt cache."""
        ts = datetime.now(timezone.utc)
        with engine.begin() as conn:
            res = conn.execute(
                text(
                    """
                    UPDATE prompts
                    SET system_prompt = :sp, user_prompt_template = :upt, updated_at = :ts
                    WHERE prompt_key = :pk
                    """
                ),
                {"sp": system_prompt or "", "upt": user_prompt_template, "ts": ts, "pk": prompt_key},
            )
            changed = getattr(res, "rowcount", None) or 0
        try:
            from shared.nlp_sql_service_v2 import invalidate_telegram_q_prompt_cache

            invalidate_telegram_q_prompt_cache()
        except Exception:
            pass
        return bool(changed)


# Initialize database on import when PostgreSQL is reachable (web_admin also calls init_db on startup)
try:
    init_db()
except Exception as e:
    logger.warning("Database init on import failed (will retry on app startup): %s", e)
