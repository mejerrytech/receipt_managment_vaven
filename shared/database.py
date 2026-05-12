import os
import json
import logging
import secrets
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, inspect, Column, Integer, String, DateTime, Text, ForeignKey, BigInteger, Float, text, or_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func

load_dotenv()

logger = logging.getLogger("database")

# Import vector service for ChromaDB integration
from shared.vector_service import get_vector_service, VectorService

# Database URL from env or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bot_data.db")

# Create engine
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def _is_postgresql() -> bool:
    return engine.dialect.name == "postgresql"


def _timestamp_column_type_sql() -> str:
    """SQLite uses DATETIME; PostgreSQL should use a proper timestamp type."""
    return "TIMESTAMP WITH TIME ZONE" if _is_postgresql() else "DATETIME"


def _table_column_names(table_name: str) -> set:
    """Return lowercase column names for SQLite / PostgreSQL (no PRAGMA — works on both)."""
    try:
        insp = inspect(engine)
        if not insp.has_table(table_name):
            return set()
        return {c["name"].lower() for c in insp.get_columns(table_name)}
    except Exception as e:
        logger.warning("Could not introspect table %s: %s", table_name, e)
        return set()


class User(Base):
    """User table - stores Telegram user details."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
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
            "id": self.id,
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
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
    document_type = Column(String(100), nullable=True)  # invoice, receipt, etc.
    
    # Key extracted fields (denormalized for easy querying)
    title = Column(String(500), nullable=True)
    document_date = Column(String(50), nullable=True)
    total_amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    vendor_name = Column(String(255), nullable=True)
    invoice_number = Column(String(100), nullable=True)
    gstin = Column(String(50), nullable=True)
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
            "id": self.id,
            "user_id": self.user_id,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "extracted_data": json.loads(self.extracted_data) if self.extracted_data else None,
            "document_type": self.document_type,
            "title": self.title,
            "document_date": self.document_date,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "vendor_name": self.vendor_name,
            "invoice_number": self.invoice_number,
            "gstin": self.gstin,
            "expense_category": self.expense_category,
            "user_input_text": self.user_input_text,
            "confidence_overall": self.confidence_overall,
            "raw_text": self.raw_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PendingDocument(Base):
    """Pending document table - stores documents awaiting user confirmation."""
    __tablename__ = "pending_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Unique token for accessing this pending document (shared across Telegram/Web)
    token = Column(String(100), unique=True, nullable=False, index=True)
    
    # Source: 'telegram' or 'web'
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
            "id": self.id,
            "token": self.token,
            "user_id": self.user_id,
            "source": self.source,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "extracted_data": json.loads(self.extracted_data) if self.extracted_data else None,
            "user_input_text": self.user_input_text,
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    intent_tag = Column(String(100), nullable=True)
    expense_category = Column(String(50), nullable=True)
    amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "intent_tag": self.intent_tag,
            "expense_category": self.expense_category,
            "amount": self.amount,
            "currency": self.currency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BotPrompt(Base):
    """Editable LLM prompts (e.g. Telegram /q pipeline)."""

    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_key = Column(String(120), unique=True, nullable=False, index=True)
    label = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False, default="telegram_q")
    system_prompt = Column(Text, nullable=False, default="")
    user_prompt_template = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def init_db():
    """Initialize database - create all tables."""
    Base.metadata.create_all(bind=engine)
    _ensure_hash_columns()
    _ensure_pending_queue_columns()
    _ensure_expense_category_columns()
    _ensure_user_input_text_columns()
    _ensure_user_text_entry_amount_columns()
    _seed_telegram_q_prompts()
    logger.info("Database initialized successfully")


def _ensure_hash_columns():
    """Lightweight migration to add hash columns on existing tables (SQLite + PostgreSQL)."""
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
    """Add expense_category column on documents for existing DBs."""
    with engine.connect() as conn:
        cols = _table_column_names("documents")
        if "expense_category" not in cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN expense_category VARCHAR(50)"))
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

    EXPENSE_CATEGORIES = [
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

    @staticmethod
    def normalize_expense_category_label(raw: Optional[str]) -> str:
        """Map any model/OCR string to a canonical value in EXPENSE_CATEGORIES."""
        c = (raw or "").strip()
        if not c:
            return "Other"
        if c in DatabaseService.EXPENSE_CATEGORIES:
            return c
        by_lower = {x.lower(): x for x in DatabaseService.EXPENSE_CATEGORIES}
        return by_lower.get(c.lower(), "Other")

    @staticmethod
    def _resolve_document_expense_category(
        data: dict,
        _merged_input_text: Optional[str],
        _file_name: Optional[str],
    ) -> str:
        """Category comes only from OCR JSON `expense_category` (normalized); no keyword tables."""
        ocr_raw = data.get("expense_category")
        if isinstance(ocr_raw, str):
            return DatabaseService.normalize_expense_category_label(ocr_raw.strip())
        return DatabaseService.normalize_expense_category_label(None)

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
    def save_document(user_id: int, file_name: Optional[str], mime_type: Optional[str],
                      file_size: Optional[int], extracted_json: str,
                      raw_text: Optional[str] = None,
                      user_input_text: Optional[str] = None,
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
            vendor = data.get("vendor_or_sender", {})
            identifiers = data.get("identifiers", {})
            confidence = data.get("confidence", {})
            merged_input_text = " ".join([x for x in [raw_text, user_input_text] if x]).strip() or None
            classified_category = DatabaseService._resolve_document_expense_category(
                data, merged_input_text, file_name
            )

            doc = Document(
                user_id=user_id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                telegram_file_unique_id=telegram_file_unique_id,
                content_sha256=content_sha256,
                dhash=dhash,
                phash=phash,
                extracted_data=extracted_json,
                document_type=data.get("document_type"),
                title=data.get("title"),
                document_date=data.get("date"),
                total_amount=amounts.get("total"),
                currency=amounts.get("currency"),
                vendor_name=vendor.get("name"),
                invoice_number=identifiers.get("invoice_number"),
                gstin=identifiers.get("gstin"),
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
        return {
            "id": doc.id,
            "user_id": doc.user_id,
            "file_name": doc.file_name,
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
            if vs.add_document(int(doc.id), int(doc.user_id), payload):
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
        for i in range(attempts):
            if vs.add_user_text_entry(
                entry_id=int(entry.id),
                user_id=int(entry.user_id),
                text=entry.text,
                intent_tag=entry.intent_tag or "expense_text",
                expense_category=entry.expense_category,
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
    def reindex_document_vector(doc_id: int, user_id: int) -> bool:
        """Reload document from SQL and push to Chroma (recovery after failed indexing)."""
        doc = DatabaseService.get_document_by_id(doc_id, user_id)
        if not doc:
            logger.warning("reindex_document_vector: no document id=%s for user_id=%s", doc_id, user_id)
            return False
        return DatabaseService.index_document_in_vector_store(doc)

    @staticmethod
    def find_documents_matching_query_tokens(user_id: int, query: str, limit: int = 10) -> List[Document]:
        """
        SQL substring fallback for /q when embedding search misses (vendor name in JSON, etc.).
        Matches vendor_name, title, raw_text, extracted_data using meaningful tokens from the question.
        """
        q = (query or "").strip()
        if len(q) < 2:
            return []
        words = re.findall(r"[A-Za-z][\w.-]{2,}", q)
        needles: List[str] = []
        if words:
            needles.append(max(words, key=len).lower())
        needles.extend([w.lower() for w in words if len(w) >= 4])
        needles = list(dict.fromkeys(needles))[:8]
        if not needles:
            needles = [q[:120].lower()]
        db = get_db()
        try:
            conds = []
            for n in needles:
                pat = f"%{n}%"
                conds.append(Document.vendor_name.ilike(pat))
                conds.append(Document.title.ilike(pat))
                conds.append(Document.raw_text.ilike(pat))
                conds.append(Document.extracted_data.ilike(pat))
            return (
                db.query(Document)
                .filter(Document.user_id == user_id)
                .filter(or_(*conds))
                .order_by(Document.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

    @staticmethod
    def save_user_text_entry(
        user_id: int,
        user_text: str,
        intent_tag: str = "expense_text",
        expense_category: Optional[str] = None,
    ) -> Optional[UserTextEntry]:
        """Persist expense-related user text and index it in vector DB."""
        db = get_db()
        try:
            cleaned = (user_text or "").strip()
            if not cleaned:
                return None
            cat_candidate = (expense_category or "").strip()
            if cat_candidate in DatabaseService.EXPENSE_CATEGORIES:
                category = cat_candidate
            else:
                category = DatabaseService.normalize_expense_category_label(cat_candidate or None)
            amount = DatabaseService._extract_amount_from_text(cleaned)
            entry = UserTextEntry(
                user_id=user_id,
                text=cleaned,
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
    def get_user_documents(user_id: int, limit: int = 50):
        """Get all documents for a user."""
        db = get_db()
        try:
            docs = db.query(Document).filter(Document.user_id == user_id)\
                     .order_by(Document.created_at.desc()).limit(limit).all()
            return docs
        finally:
            db.close()

    @staticmethod
    def get_document_by_id(doc_id: int, user_id: int) -> Optional[Document]:
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
    def fetch_documents_for_vector_enrichment(user_id: int, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch-load documents for semantic / vector search enrichment (Postgres or SQLite)."""
        if not doc_ids:
            return {}
        db = get_db()
        try:
            docs = (
                db.query(Document)
                .filter(Document.user_id == user_id, Document.id.in_(doc_ids))
                .all()
            )
            out: Dict[int, Dict[str, Any]] = {}
            for d in docs:
                out[d.id] = {
                    "id": d.id,
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
        user_id: int, entry_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
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
            out: Dict[int, Dict[str, Any]] = {}
            for t in rows:
                out[t.id] = {
                    "id": t.id,
                    "text": t.text,
                    "amount": t.amount,
                    "currency": t.currency,
                    "expense_category": t.expense_category,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
            return out
        finally:
            db.close()

    @staticmethod
    def get_distinct_vendor_names(user_id: int, limit: int = 500) -> List[str]:
        """Distinct vendor names for a user (helper for NLP user_info replies)."""
        db = get_db()
        try:
            rows = (
                db.query(Document.vendor_name)
                .filter(Document.user_id == user_id, Document.vendor_name.isnot(None))
                .distinct()
                .limit(limit)
                .all()
            )
            return [r[0] for r in rows if r and r[0]]
        finally:
            db.close()

    @staticmethod
    def get_user_summary_stats(user_id: int) -> dict:
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
            vendor_count = db.query(func.count(func.distinct(Document.vendor_name))).filter(
                Document.user_id == user_id,
                Document.vendor_name.isnot(None)
            ).scalar() or 0

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
                        "id": d.id,
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
    def create_pending_document(user_id: int, file_name: Optional[str], mime_type: Optional[str],
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
                confidence_overall=confidence_overall,
                telegram_chat_id=telegram_chat_id,
                telegram_message_id=telegram_message_id,
                telegram_file_id=telegram_file_id,
                status=status,
                ocr_job_id=ocr_job_id,
                ocr_started_at=datetime.utcnow() if status == 'processing' else None,
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
    def get_pending_document_by_id(pending_id: int, user_id: int) -> Optional[PendingDocument]:
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
    def get_pending_document_for_job(pending_id: int) -> Optional[PendingDocument]:
        """Get pending document for background processing regardless of user/session."""
        db = get_db()
        try:
            return db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
        finally:
            db.close()

    @staticmethod
    def get_user_pending_documents(user_id: int, limit: int = 20):
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
    def confirm_pending_document(pending_id: int) -> Optional[Document]:
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
    def update_pending_document(pending_id: int, updated_json: str) -> Optional[PendingDocument]:
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
    def count_user_inflight_pending_documents(user_id: int) -> int:
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
    def set_pending_job_id(pending_id: int, job_id: str) -> Optional[PendingDocument]:
        """Attach Celery job id to pending document."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
            if not pending:
                return None
            pending.ocr_job_id = job_id
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
    def set_pending_telegram_message_id(pending_id: int, message_id: int) -> Optional[PendingDocument]:
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
    def mark_pending_ocr_ready(pending_id: int, extracted_json: str, confidence_overall: Optional[float]) -> Optional[PendingDocument]:
        """Mark pending OCR job complete and ready for user confirmation."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(PendingDocument.id == pending_id).first()
            if not pending or pending.status in ('confirmed', 'cancelled'):
                return None
            pending.extracted_data = extracted_json
            pending.confidence_overall = confidence_overall
            pending.status = 'ready'
            pending.error_message = None
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
    def mark_pending_ocr_failed(pending_id: int, error_message: str, retry_count: int = 0) -> Optional[PendingDocument]:
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
        identifiers = data.get("identifiers") or {}

        total_amount = amounts.get("total")
        try:
            total_amount = float(total_amount) if total_amount is not None else None
        except Exception:
            total_amount = None

        return {
            "vendor_name": DatabaseService._normalize_text(vendor.get("name")),
            "invoice_number": DatabaseService._normalize_text(identifiers.get("invoice_number")),
            "date": DatabaseService._normalize_text(data.get("date")),
            "title": DatabaseService._normalize_text(data.get("title")),
            "total_amount": total_amount
        }

    @staticmethod
    def find_duplicate_image_for_user(user_id: int,
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
                        "id": d.id,
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
    def find_duplicate_by_extracted_fingerprint(user_id: int, extracted_json: str) -> Optional[dict]:
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
                    return {"source": "documents", "id": d.id, "file_name": d.file_name, "match_type": "ocr_fingerprint_invoice"}
                if vendor and e_vendor and vendor == e_vendor and amount_match and date and e_date and date == e_date:
                    return {"source": "documents", "id": d.id, "file_name": d.file_name, "match_type": "ocr_fingerprint_vendor_amount_date"}
                if title and e_title and title == e_title and amount_match and date and e_date and date == e_date:
                    return {"source": "documents", "id": d.id, "file_name": d.file_name, "match_type": "ocr_fingerprint_title_amount_date"}

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
    def cancel_pending_document(pending_id: int) -> bool:
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


# Initialize database on module import
init_db()
