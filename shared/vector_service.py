"""
Vector Database Service using ChromaDB for semantic search fallback.

This service provides:
- Document embedding and storage in ChromaDB
- Semantic search with user_id guardrails (security filtering)
- Fallback mechanism when SQL queries fail
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from functools import lru_cache

load_dotenv()

logger = logging.getLogger("vector_service")

# ChromaDB storage path
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Simple LRU cache for embeddings (text_hash -> embedding vector)
_embedding_cache: Dict[str, List[float]] = {}
_MAX_CACHE_SIZE = 1000


class VectorService:
    """Service for vector-based semantic search using ChromaDB."""

    def __init__(self):
        # Initialize OpenAI client for embeddings
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.error("OPENAI_API_KEY not set!")
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.embedding_model = OPENAI_EMBEDDING_MODEL

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection for documents
        # Note: If embedding dimensions change, collection will be recreated in add_document
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"VectorService initialized with OpenAI embeddings and ChromaDB at {CHROMA_DB_PATH}")

    def _generate_embedding(self, text: str, max_retries: int = 4) -> List[float]:
        """Generate embedding for text using OpenAI with caching and transient-error retries."""
        import hashlib
        global _embedding_cache

        cache_key = hashlib.md5(text[:8000].encode()).hexdigest()

        if cache_key in _embedding_cache:
            logger.debug(f"Embedding cache hit for key {cache_key[:8]}")
            return _embedding_cache[cache_key]

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text[:8000]
                )
                embedding = response.data[0].embedding

                if len(_embedding_cache) < _MAX_CACHE_SIZE:
                    _embedding_cache[cache_key] = embedding

                return embedding
            except Exception as e:
                last_err = e
                logger.warning(
                    "Embedding attempt %s/%s failed: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(min(2.0, 0.4 * (2**attempt)))

        logger.error(f"Failed to generate embedding after {max_retries} attempts: {last_err}")
        raise last_err if last_err else RuntimeError("embedding failed")

    def _flatten_json_for_search(self, data: Any, prefix: str = "") -> List[str]:
        """Recursively flatten JSON so nested fields (e.g. address/location) are searchable."""
        parts: List[str] = []
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                parts.extend(self._flatten_json_for_search(value, new_prefix))
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                new_prefix = f"{prefix}[{idx}]"
                parts.extend(self._flatten_json_for_search(value, new_prefix))
        elif data is not None:
            parts.append(f"{prefix}: {data}")
        return parts

    def _document_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document data to searchable text."""
        parts = []

        # Add title
        if doc.get('title'):
            parts.append(f"Title: {doc['title']}")

        if doc.get('expense_category'):
            parts.append(f"Category: {doc['expense_category']}")

        # Add document type
        if doc.get('document_type'):
            parts.append(f"Type: {doc['document_type']}")

        # Add vendor
        if doc.get('vendor_name'):
            parts.append(f"Vendor: {doc['vendor_name']}")

        # Add amount
        if doc.get('total_amount'):
            parts.append(f"Amount: {doc['total_amount']}")

        # Add extracted JSON content if available (including nested fields)
        if doc.get('extracted_json'):
            try:
                data = json.loads(doc['extracted_json'])
                flattened = self._flatten_json_for_search(data)
                # Skip noisy OCR confidence/table blobs but keep business fields like address/location/items.
                flattened = [
                    line for line in flattened
                    if not line.startswith("confidence") and not line.startswith("tables")
                ]
                parts.extend(flattened)
            except json.JSONDecodeError:
                pass

        return " | ".join(parts) if parts else "Untitled Document"

    def add_document(self, doc_id: int, user_id: int, doc_data: Dict[str, Any]) -> bool:
        """
        Add or update a document in the vector database.

        Args:
            doc_id: Document database ID
            user_id: User ID (for guardrails)
            doc_data: Document data dictionary

        Returns:
            True if successful
        """
        try:
            # Convert document to searchable text
            text = self._document_to_text(doc_data)

            # Generate embedding
            embedding = self._generate_embedding(text)

            ec = doc_data.get("expense_category")
            uid = int(user_id)
            did = int(doc_id)
            meta_base = {
                "user_id": uid,
                "doc_id": did,
                "text": text[:1000],
            }
            if ec:
                meta_base["expense_category"] = str(ec)[:80]

            # Try to add to ChromaDB
            try:
                self.collection.upsert(
                    ids=[str(doc_id)],
                    embeddings=[embedding],
                    metadatas=[meta_base],
                    documents=[text]
                )
            except Exception as e:
                # If dimension mismatch, recreate collection
                if "dimension" in str(e).lower():
                    logger.warning(f"Embedding dimension mismatch, recreating collection: {e}")
                    collection_name = self.collection.name
                    self.chroma_client.delete_collection(name=collection_name)
                    self.collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    # Retry adding document
                    self.collection.upsert(
                        ids=[str(doc_id)],
                        embeddings=[embedding],
                        metadatas=[meta_base],
                        documents=[text]
                    )
                else:
                    raise

            logger.info(f"Added document {doc_id} to vector DB for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to vector DB: {e}")
            return False

    def add_user_text_entry(
        self,
        entry_id: int,
        user_id: int,
        text: str,
        intent_tag: str = "expense_text",
        expense_category: Optional[str] = None,
    ) -> bool:
        """Add user free-text entry to vector DB for semantic recall."""
        try:
            normalized_text = (text or "").strip()
            if not normalized_text:
                return False

            embedding = self._generate_embedding(normalized_text)
            vector_id = f"text_entry_{entry_id}"
            meta = {
                "user_id": int(user_id),
                "doc_id": -int(entry_id),
                "type": "user_text_entry",
                "intent_tag": intent_tag,
                "text": normalized_text[:1000],
            }
            if expense_category:
                meta["expense_category"] = str(expense_category)[:80]

            self.collection.upsert(
                ids=[vector_id],
                embeddings=[embedding],
                metadatas=[meta],
                documents=[normalized_text]
            )
            logger.info(f"Added user text entry {entry_id} to vector DB for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add user text entry {entry_id} to vector DB: {e}")
            return False

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document from the vector database."""
        try:
            self.collection.delete(ids=[str(doc_id)])
            logger.info(f"Deleted document {doc_id} from vector DB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def search(
        self,
        query: str,
        user_id: int,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by semantic similarity.

        SECURITY: Always filters by user_id - users can only see their own documents.

        Args:
            query: Search query text
            user_id: User ID to filter by (mandatory guardrail)
            n_results: Number of results to return

        Returns:
            List of matching documents with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Search with user_id filter in metadata
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where={"user_id": int(user_id)},
                    include=["metadatas", "documents", "distances"]
                )
            except Exception as e:
                # If dimension mismatch, recreate collection and return empty results
                if "dimension" in str(e).lower():
                    logger.warning(f"Embedding dimension mismatch in search, recreating collection: {e}")
                    collection_name = self.collection.name
                    self.chroma_client.delete_collection(name=collection_name)
                    self.collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("Collection recreated with new embedding dimensions")
                    return []
                else:
                    raise

            # Format results
            matches = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    document = results['documents'][0][i] if results['documents'] else ""
                    distance = results['distances'][0][i] if results['distances'] else 1.0

                    # Calculate similarity score (0-100%)
                    similarity = max(0, min(100, (1 - distance) * 100))

                    entry_type = metadata.get("type", "document")

                    # Skip query-history artifacts.
                    if isinstance(doc_id, str) and doc_id.startswith("sql_result_"):
                        continue

                    if entry_type == "user_text_entry" and isinstance(doc_id, str) and doc_id.startswith("text_entry_"):
                        try:
                            text_entry_id = int(doc_id.split("text_entry_")[1])
                        except Exception:
                            continue
                        matches.append({
                            "doc_id": None,
                            "text_entry_id": text_entry_id,
                            "entry_type": "user_text_entry",
                            "user_id": metadata.get("user_id", user_id),
                            "text": document[:500],
                            "expense_category": metadata.get("expense_category"),
                            "similarity_score": round(similarity, 2),
                            "source": "vector_search"
                        })
                        continue

                    # Regular document IDs
                    try:
                        doc_id_int = int(doc_id)
                    except (ValueError, TypeError):
                        continue

                    matches.append({
                        "doc_id": doc_id_int,
                        "text_entry_id": None,
                        "entry_type": "document",
                        "user_id": metadata.get("user_id", user_id),
                        "text": document[:500],  # Truncate for display
                        "expense_category": metadata.get("expense_category"),
                        "similarity_score": round(similarity, 2),
                        "source": "vector_search"
                    })

            logger.info(f"Vector search for user {user_id}: found {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Vector search failed for user {user_id}: {e}")
            return []

    def get_similar_documents(
        self,
        doc_id: int,
        user_id: int,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.

        Args:
            doc_id: Reference document ID
            user_id: User ID (guardrail)
            n_results: Number of similar documents to return

        Returns:
            List of similar documents
        """
        try:
            # Get the reference document embedding
            result = self.collection.get(
                ids=[str(doc_id)],
                include=["embeddings"]
            )

            if not result['embeddings']:
                return []

            # Find similar documents with user_id filter
            similar = self.collection.query(
                query_embeddings=result['embeddings'],
                n_results=n_results + 1,  # +1 to exclude the document itself
                where={"user_id": int(user_id)},
                include=["metadatas", "documents", "distances"]
            )

            matches = []
            if similar['ids'] and similar['ids'][0]:
                for i, sid in enumerate(similar['ids'][0]):
                    try:
                        sid_int = int(sid)
                    except (ValueError, TypeError):
                        # Skip non-numeric IDs (SQL result entries)
                        continue

                    if sid_int != doc_id:  # Exclude the reference document
                        metadata = similar['metadatas'][0][i] if similar['metadatas'] else {}
                        document = similar['documents'][0][i] if similar['documents'] else ""
                        distance = similar['distances'][0][i] if similar['distances'] else 1.0

                        similarity = max(0, min(100, (1 - distance) * 100))

                        matches.append({
                            "doc_id": sid_int,
                            "text": document[:500],
                            "similarity_score": round(similarity, 2)
                        })

            return matches[:n_results]

        except Exception as e:
            logger.error(f"Failed to get similar documents for {doc_id}: {e}")
            return []


# Singleton instance
_vector_service = None


def get_vector_service() -> VectorService:
    """Get or create the vector service singleton."""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
