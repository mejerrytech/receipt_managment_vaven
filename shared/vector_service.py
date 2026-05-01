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
from typing import List, Dict, Any, Optional
from google import genai
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from functools import lru_cache

load_dotenv()

logger = logging.getLogger("vector_service")

# ChromaDB storage path
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Gemini embedding model
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Simple LRU cache for embeddings (text_hash -> embedding vector)
_embedding_cache: Dict[str, List[float]] = {}
_MAX_CACHE_SIZE = 1000


class VectorService:
    """Service for vector-based semantic search using ChromaDB."""

    def __init__(self):
        # Initialize Gemini client for embeddings
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embedding_model = GEMINI_EMBEDDING_MODEL

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

        logger.info(f"VectorService initialized with Gemini embeddings and ChromaDB at {CHROMA_DB_PATH}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini with caching."""
        import hashlib
        global _embedding_cache

        # Create cache key from text hash
        cache_key = hashlib.md5(text[:8000].encode()).hexdigest()

        # Check cache first
        if cache_key in _embedding_cache:
            logger.debug(f"Embedding cache hit for key {cache_key[:8]}")
            return _embedding_cache[cache_key]

        try:
            response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=text[:8000]
            )
            embedding = response.embeddings[0].values

            # Store in cache (with size limit)
            if len(_embedding_cache) < _MAX_CACHE_SIZE:
                _embedding_cache[cache_key] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _document_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document data to searchable text."""
        parts = []

        # Add title
        if doc.get('title'):
            parts.append(f"Title: {doc['title']}")

        # Add document type
        if doc.get('document_type'):
            parts.append(f"Type: {doc['document_type']}")

        # Add vendor
        if doc.get('vendor_name'):
            parts.append(f"Vendor: {doc['vendor_name']}")

        # Add amount
        if doc.get('total_amount'):
            parts.append(f"Amount: {doc['total_amount']}")

        # Add extracted JSON content if available
        if doc.get('extracted_json'):
            try:
                data = json.loads(doc['extracted_json'])
                # Flatten JSON to text
                for key, value in data.items():
                    if isinstance(value, (str, int, float)) and key not in ['tables', 'confidence']:
                        parts.append(f"{key}: {value}")
                    elif isinstance(value, list) and key == 'line_items':
                        for item in value:
                            if isinstance(item, dict):
                                item_text = " | ".join([f"{k}: {v}" for k, v in item.items()])
                                parts.append(f"Item: {item_text}")
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

            # Try to add to ChromaDB
            try:
                self.collection.upsert(
                    ids=[str(doc_id)],
                    embeddings=[embedding],
                    metadatas=[{
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "text": text[:1000]  # Store truncated text for display
                    }],
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
                        metadatas=[{
                            "user_id": user_id,
                            "doc_id": doc_id,
                            "text": text[:1000]  # Store truncated text for display
                        }],
                        documents=[text]
                    )
                else:
                    raise

            logger.info(f"Added document {doc_id} to vector DB for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to vector DB: {e}")
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
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": user_id},  # 🔐 GUARDRAIL: Only user's own documents
                include=["metadatas", "documents", "distances"]
            )

            # Format results
            matches = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    document = results['documents'][0][i] if results['documents'] else ""
                    distance = results['distances'][0][i] if results['distances'] else 1.0

                    # Calculate similarity score (0-100%)
                    similarity = max(0, min(100, (1 - distance) * 100))

                    # Handle both numeric document IDs and string SQL result IDs
                    try:
                        # Try to convert to int (regular document ID)
                        doc_id_int = int(doc_id)
                    except (ValueError, TypeError):
                        # Skip SQL result entries - they have string IDs like "sql_result_..."
                        # These are query history, not actual documents
                        continue

                    matches.append({
                        "doc_id": doc_id_int,
                        "user_id": metadata.get("user_id", user_id),
                        "text": document[:500],  # Truncate for display
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
                where={"user_id": user_id},  # 🔐 GUARDRAIL
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
