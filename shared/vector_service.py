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
import math
import time
import uuid
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from functools import lru_cache

from shared.id_types import as_str

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
        # SQL query cache lives in a separate collection so ephemeral ids never corrupt document HNSW.
        self.sql_cache_collection = self.chroma_client.get_or_create_collection(
            name="sql_query_cache",
            metadata={"hnsw:space": "cosine"},
        )

        self._purge_legacy_sql_result_rows()

        logger.info(f"VectorService initialized with OpenAI embeddings and ChromaDB at {CHROMA_DB_PATH}")

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _format_search_matches(
        self,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        distances: List[float],
        user_id: uuid.UUID,
        username: Optional[str],
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        expected_username = (username or "").strip()
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            distance = distances[i] if i < len(distances) else 1.0

            metadata_user_id = metadata.get("user_id", as_str(user_id))
            if str(metadata_user_id) != str(user_id):
                continue

            metadata_username = (metadata.get("username") or "").strip()
            if expected_username and metadata_username and metadata_username != expected_username:
                continue

            similarity = max(0, min(100, (1 - distance) * 100))
            entry_type = metadata.get("type", "document")

            if isinstance(doc_id, str) and doc_id.startswith("sql_result_"):
                continue

            if entry_type == "user_text_entry" and isinstance(doc_id, str) and doc_id.startswith("text_entry_"):
                text_entry_id = doc_id.split("text_entry_", 1)[1]
                if not text_entry_id:
                    continue
                matches.append({
                    "doc_id": None,
                    "text_entry_id": text_entry_id,
                    "entry_type": "user_text_entry",
                    "user_id": metadata_user_id,
                    "username": metadata_username or expected_username,
                    "text": document[:500],
                    "expense_category": metadata.get("expense_category"),
                    "similarity_score": round(similarity, 2),
                    "source": "vector_search",
                })
                continue

            doc_id_value = metadata.get("doc_id") or doc_id
            if not doc_id_value or str(doc_id_value).startswith("pending_"):
                continue

            matches.append({
                "doc_id": str(doc_id_value),
                "text_entry_id": None,
                "entry_type": "document",
                "user_id": metadata_user_id,
                "username": metadata_username or expected_username,
                "text": document[:500],
                "expense_category": metadata.get("expense_category"),
                "similarity_score": round(similarity, 2),
                "source": "vector_search",
            })
        return matches

    def _search_bruteforce(
        self,
        query_embedding: List[float],
        user_id: uuid.UUID,
        n_results: int,
        username: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fallback when Chroma HNSW query fails: scan user rows and rank by cosine similarity."""
        uid = as_str(user_id)
        try:
            batch = self.collection.get(
                where={"user_id": uid},
                include=["metadatas", "documents", "embeddings"],
            )
        except Exception as e:
            logger.error("Bruteforce vector fallback get() failed for user %s: %s", uid, e)
            return []

        ids = batch.get("ids") or []
        metadatas = batch.get("metadatas") or []
        documents = batch.get("documents") or []
        embeddings = batch.get("embeddings") or []
        if not ids or not embeddings:
            return []

        scored: List[tuple[float, int]] = []
        for idx, emb in enumerate(embeddings):
            if not emb:
                continue
            vid = ids[idx]
            if isinstance(vid, str) and vid.startswith("sql_result_"):
                continue
            scored.append((self._cosine_similarity(query_embedding, emb), idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(n_results, 1)]
        if not top:
            return []

        out_ids: List[str] = []
        out_meta: List[Dict[str, Any]] = []
        out_docs: List[str] = []
        out_dist: List[float] = []
        for sim, idx in top:
            out_ids.append(ids[idx])
            out_meta.append(metadatas[idx] if idx < len(metadatas) else {})
            out_docs.append(documents[idx] if idx < len(documents) else "")
            out_dist.append(max(0.0, 1.0 - sim))

        matches = self._format_search_matches(
            out_ids, out_meta, out_docs, out_dist, user_id, username
        )
        logger.info(
            "Bruteforce vector fallback for user %s: found %s matches",
            uid,
            len(matches),
        )
        return matches

    def _repair_collection_index(self) -> None:
        """Recreate collection when Chroma HNSW index is corrupted."""
        collection_name = self.collection.name
        logger.warning("Recreating Chroma collection %s due to index corruption", collection_name)
        self.chroma_client.delete_collection(name=collection_name)
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _purge_legacy_sql_result_rows(self) -> int:
        """Remove sql_result_* rows accidentally stored in the documents collection."""
        try:
            batch = self.collection.get(include=[])
            ids = [
                row_id for row_id in (batch.get("ids") or [])
                if isinstance(row_id, str) and row_id.startswith("sql_result_")
            ]
            if ids:
                self.collection.delete(ids=ids)
                logger.info("Purged %s legacy sql_result rows from documents collection", len(ids))
            return len(ids)
        except Exception as e:
            logger.warning("Could not purge legacy sql_result rows: %s", e)
            return 0

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

    _FLATTEN_SKIP_PREFIXES = (
        "confidence.",
        "display_card",
        "_ocr_metadata",
    )

    def _document_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document data to rich searchable text for embeddings (full OCR payload)."""
        parts = []

        if doc.get("title"):
            parts.append(f"Title: {doc['title']}")

        if doc.get("expense_category"):
            parts.append(f"Category: {doc['expense_category']}")

        if doc.get("document_type"):
            parts.append(f"Type: {doc['document_type']}")

        if doc.get("vendor_name"):
            parts.append(f"Vendor: {doc['vendor_name']}")

        if doc.get("total_amount"):
            parts.append(f"Amount: {doc['total_amount']}")

        if doc.get("document_date"):
            parts.append(f"Receipt date: {doc['document_date']}")

        if doc.get("user_input_text"):
            parts.append(f"User note: {doc['user_input_text']}")

        if doc.get("status") == "pending":
            parts.append("Status: pending (not yet confirmed)")

        raw = (doc.get("raw_text") or "").strip()
        if raw:
            parts.append(f"Raw receipt OCR: {raw}")

        extracted = doc.get("extracted_json")
        if extracted:
            try:
                data = json.loads(extracted) if isinstance(extracted, str) else extracted
                if isinstance(data, dict):
                    flattened = self._flatten_json_for_search(data)
                    flattened = [
                        line
                        for line in flattened
                        if not any(line.startswith(prefix) for prefix in self._FLATTEN_SKIP_PREFIXES)
                    ]
                    parts.extend(flattened)
                    parts.append(
                        "Full OCR JSON: "
                        + json.dumps(data, ensure_ascii=False, default=str)
                    )
                elif data is not None:
                    parts.append(f"Extracted data: {data}")
            except (json.JSONDecodeError, TypeError):
                if isinstance(extracted, str) and extracted.strip():
                    parts.append(f"Extracted OCR text: {extracted.strip()}")

        text = " | ".join(parts) if parts else "Untitled Document"
        return text[:8000]

    def add_document(self, doc_id: uuid.UUID, user_id: uuid.UUID, doc_data: Dict[str, Any]) -> bool:
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
            uid = as_str(user_id)
            did = as_str(doc_id)
            meta_base = {
                "user_id": uid,
                "doc_id": did,
                "text": text[:1000],
            }
            username = (doc_data.get("username") or "").strip()
            if username:
                meta_base["username"] = username[:255]
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

    def add_pending_document(self, pending_id: uuid.UUID, user_id: uuid.UUID, doc_data: Dict[str, Any]) -> bool:
        """Embed OCR-extracted pending upload so Q&A works before CONFIRM."""
        try:
            payload = {**doc_data, "status": "pending"}
            text = self._document_to_text(payload)
            embedding = self._generate_embedding(text)
            vector_id = f"pending_{pending_id}"
            uid = as_str(user_id)
            pid = as_str(pending_id)
            meta_base = {
                "user_id": uid,
                "pending_id": pid,
                "doc_id": pid,
                "type": "pending_document",
                "text": text[:1000],
            }
            username = (doc_data.get("username") or "").strip()
            if username:
                meta_base["username"] = username[:255]
            ec = doc_data.get("expense_category")
            if ec:
                meta_base["expense_category"] = str(ec)[:80]

            self.collection.upsert(
                ids=[vector_id],
                embeddings=[embedding],
                metadatas=[meta_base],
                documents=[text],
            )
            logger.info("Added pending document %s to vector DB for user %s", pending_id, user_id)
            return True
        except Exception as e:
            logger.error("Failed to add pending document %s to vector DB: %s", pending_id, e)
            return False

    def delete_pending_document(self, pending_id: uuid.UUID) -> bool:
        """Remove pending OCR embedding after confirm or discard."""
        try:
            self.collection.delete(ids=[f"pending_{pending_id}"])
            logger.info("Deleted pending document %s from vector DB", pending_id)
            return True
        except Exception as e:
            logger.error("Failed to delete pending document %s from vector DB: %s", pending_id, e)
            return False

    def add_user_text_entry(
        self,
        entry_id: uuid.UUID,
        user_id: uuid.UUID,
        text: str,
        intent_tag: str = "expense_text",
        expense_category: Optional[str] = None,
        username: Optional[str] = None,
    ) -> bool:
        """Add user free-text entry to vector DB for semantic recall."""
        try:
            normalized_text = (text or "").strip()
            if not normalized_text:
                return False

            embedding = self._generate_embedding(normalized_text)
            vector_id = f"text_entry_{entry_id}"
            meta = {
                "user_id": as_str(user_id),
                "doc_id": as_str(entry_id),
                "type": "user_text_entry",
                "intent_tag": intent_tag,
                "text": normalized_text[:1000],
            }
            normalized_username = (username or "").strip()
            if normalized_username:
                meta["username"] = normalized_username[:255]
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

    def delete_document(self, doc_id: uuid.UUID) -> bool:
        """Delete a document from the vector database."""
        try:
            self.collection.delete(ids=[str(doc_id)])
            logger.info(f"Deleted document {doc_id} from vector DB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def purge_user_vectors(self, user_id: uuid.UUID) -> int:
        """Remove all Chroma rows for one user (user_id metadata guardrail)."""
        uid = as_str(user_id)
        try:
            existing = self.collection.get(where={"user_id": uid}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
                logger.info("Purged %s vector rows for user_id=%s", len(ids), uid)
            return len(ids)
        except Exception as e:
            logger.error("Failed to purge vectors for user %s: %s", uid, e)
            return 0

    def search(
        self,
        query: str,
        user_id: uuid.UUID,
        n_results: int = 10,
        username: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents by semantic similarity.

        SECURITY: Always filters by user_id and drops any username metadata mismatch.

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
                    where={"user_id": as_str(user_id)},
                    include=["metadatas", "documents", "distances"]
                )
            except Exception as e:
                err = str(e).lower()
                # If dimension mismatch, recreate collection and return empty results
                if "dimension" in err:
                    logger.warning(f"Embedding dimension mismatch in search, recreating collection: {e}")
                    self._repair_collection_index()
                    logger.info("Collection recreated with new embedding dimensions")
                    return []
                if "finding id" in err or "executing plan" in err:
                    logger.warning("Chroma index query failed (%s); trying bruteforce fallback", e)
                    return self._search_bruteforce(
                        query_embedding, user_id, n_results, username
                    )
                raise

            # Format results
            if results['ids'] and results['ids'][0]:
                matches = self._format_search_matches(
                    results['ids'][0],
                    results['metadatas'][0] if results['metadatas'] else [],
                    results['documents'][0] if results['documents'] else [],
                    results['distances'][0] if results['distances'] else [],
                    user_id,
                    username,
                )
            else:
                matches = []

            logger.info(f"Vector search for user {user_id}: found {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Vector search failed for user {user_id}: {e}")
            try:
                query_embedding = self._generate_embedding(query)
                return self._search_bruteforce(
                    query_embedding, user_id, n_results, username
                )
            except Exception as fallback_err:
                logger.error(
                    "Vector search bruteforce fallback failed for user %s: %s",
                    user_id,
                    fallback_err,
                )
                return []

    def get_similar_documents(
        self,
        doc_id: uuid.UUID,
        user_id: uuid.UUID,
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
                where={"user_id": as_str(user_id)},
                include=["metadatas", "documents", "distances"]
            )

            matches = []
            if similar['ids'] and similar['ids'][0]:
                for i, sid in enumerate(similar['ids'][0]):
                    if str(sid) == str(doc_id):
                        continue

                    metadata = similar['metadatas'][0][i] if similar['metadatas'] else {}
                    document = similar['documents'][0][i] if similar['documents'] else ""
                    distance = similar['distances'][0][i] if similar['distances'] else 1.0

                    similarity = max(0, min(100, (1 - distance) * 100))

                    matches.append({
                        "doc_id": str(metadata.get("doc_id") or sid),
                        "text": document[:500],
                        "similarity_score": round(similarity, 2),
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
