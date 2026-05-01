import os
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, BigInteger, Float
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
    
    # OCR extracted data (stored as JSON string)
    extracted_data = Column(Text, nullable=True)
    
    # Confidence score
    confidence_overall = Column(Float, nullable=True)
    
    # Telegram-specific fields (for callback handling)
    telegram_chat_id = Column(BigInteger, nullable=True)
    telegram_message_id = Column(Integer, nullable=True)
    
    # Status: 'pending', 'confirmed', 'cancelled'
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
            "confidence_overall": self.confidence_overall,
            "telegram_chat_id": self.telegram_chat_id,
            "telegram_message_id": self.telegram_message_id,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


def init_db():
    """Initialize database - create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


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
                      raw_text: Optional[str] = None) -> Document:
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

            doc = Document(
                user_id=user_id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                extracted_data=extracted_json,
                document_type=data.get("document_type"),
                title=data.get("title"),
                document_date=data.get("date"),
                total_amount=amounts.get("total"),
                currency=amounts.get("currency"),
                vendor_name=vendor.get("name"),
                invoice_number=identifiers.get("invoice_number"),
                gstin=identifiers.get("gstin"),
                confidence_overall=confidence.get("overall"),
                raw_text=raw_text or data.get("text_content")
            )
            
            db.add(doc)
            db.commit()
            db.refresh(doc)
            logger.info(f"Saved document: {doc}")

            # Also add to vector database for semantic search (fire and forget)
            try:
                vector_service = get_vector_service()
                doc_dict = {
                    "id": doc.id,
                    "user_id": doc.user_id,
                    "file_name": doc.file_name,
                    "document_type": doc.document_type,
                    "title": doc.title,
                    "vendor_name": doc.vendor_name,
                    "total_amount": doc.total_amount,
                    "extracted_json": doc.extracted_data,
                    "document_date": doc.document_date,
                    "raw_text": doc.raw_text
                }
                vector_service.add_document(doc.id, doc.user_id, doc_dict)
            except Exception as e:
                logger.warning(f"Failed to add document {doc.id} to vector DB (non-critical): {e}")

            return doc
        except Exception as e:
            db.rollback()
            logger.error(f"Database error in save_document: {e}")
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
                                source: str = 'web',
                                telegram_chat_id: Optional[int] = None,
                                telegram_message_id: Optional[int] = None) -> PendingDocument:
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
                extracted_data=extracted_json,
                confidence_overall=confidence_overall,
                telegram_chat_id=telegram_chat_id,
                telegram_message_id=telegram_message_id,
                status='pending',
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
                PendingDocument.status == 'pending'
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
                PendingDocument.status == 'pending'
            ).first()
        finally:
            db.close()

    @staticmethod
    def get_user_pending_documents(user_id: int, limit: int = 20):
        """Get all pending documents for a user."""
        db = get_db()
        try:
            pendings = db.query(PendingDocument).filter(
                PendingDocument.user_id == user_id,
                PendingDocument.status == 'pending'
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
                PendingDocument.status == 'pending'
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
                extracted_json=pending.extracted_data
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
                PendingDocument.status == 'pending'
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
    def cancel_pending_document(pending_id: int) -> bool:
        """Cancel a pending document."""
        db = get_db()
        try:
            pending = db.query(PendingDocument).filter(
                PendingDocument.id == pending_id,
                PendingDocument.status == 'pending'
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
                PendingDocument.status == 'pending',
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


# Initialize database on module import
init_db()
