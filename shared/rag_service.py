"""
RAG Service - Retrieval Augmented Generation using GPT-4o with Function Calling

This service provides:
- Semantic document retrieval
- GPT-4o powered document analysis with function calling
- Anthropic fallback when needed
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from shared.orchestrator import get_orchestrator, AgentType, ModelProvider
from shared.vector_service import get_vector_service

load_dotenv()

logger = logging.getLogger("rag_service")

# Function schemas for RAG
SEARCH_DOCUMENTS_FUNCTION = {
    "name": "search_documents",
    "description": "Search for documents in the vector database by semantic similarity",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents"
            },
            "n_results": {
                "type": "integer",
                "description": "Number of documents to retrieve (default 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

GET_DOCUMENT_DETAILS_FUNCTION = {
    "name": "get_document_details",
    "description": "Get full details of specific documents from the database",
    "parameters": {
        "type": "object",
        "properties": {
            "doc_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of document IDs to retrieve"
            }
        },
        "required": ["doc_ids"]
    }
}

ANSWER_QUESTION_FUNCTION = {
    "name": "answer_question",
    "description": "Provide a natural language answer based on retrieved documents",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The natural language answer to the user's question"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1",
                "minimum": 0,
                "maximum": 1
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of document references/citations"
            }
        },
        "required": ["answer", "confidence"]
    }
}


class RAGService:
    """
    RAG Service using GPT-4o with Function Calling via Orchestration Layer.

    Architecture:
    1. Semantic Search (Vector DB)
    2. GPT-4o Analysis with function calling (search, retrieve, answer)
    3. Anthropic fallback if needed
    """

    def __init__(self):
        self.orchestrator = get_orchestrator()
        self.vector_service = get_vector_service()

        logger.info("RAGService initialized with orchestration layer")

    def answer_question(
        self,
        user_query: str,
        user_id: int,
        n_results: int = 5,
        db_path: str = "bot_data.db"
    ) -> Dict[str, Any]:
        """
        Answer a user question using RAG approach.

        Pipeline:
        1. Retrieve relevant documents via semantic search
        2. Use GPT-4o to analyze and answer
        """
        # Step 1: Semantic Search
        retrieved_docs = self._retrieve_documents(user_query, user_id, n_results)

        if not retrieved_docs:
            return {
                "success": False,
                "answer": "I couldn't find any relevant documents to answer your question.",
                "documents": [],
                "confidence": 0,
                "error": "No documents found"
            }

        # Step 2: Enrich documents with full details
        enriched_docs = self._enrich_documents(retrieved_docs, user_id, db_path)

        # Step 3: Generate answer using GPT-4o with function calling
        answer_result = self._generate_answer(user_query, enriched_docs)

        return {
            "success": True,
            "answer": answer_result.get("answer", ""),
            "documents": enriched_docs,
            "confidence": answer_result.get("confidence", 0),
            "citations": answer_result.get("citations", []),
            "provider_used": answer_result.get("provider_used", "unknown"),
            "retrieved_count": len(retrieved_docs)
        }

    def _retrieve_documents(
        self,
        query: str,
        user_id: int,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve documents via semantic search."""
        try:
            results = self.vector_service.search(
                query=query,
                user_id=user_id,
                n_results=n_results
            )
            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []

    def _enrich_documents(
        self,
        docs: List[Dict],
        user_id: int,
        db_path: str
    ) -> List[Dict[str, Any]]:
        """Enrich vector search results with full DB details."""
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            doc_ids = [d["doc_id"] for d in docs]
            placeholders = ",".join(["?"] * len(doc_ids))

            cursor.execute(f"""
                SELECT id, document_type, title, total_amount, vendor_name,
                       invoice_number, currency, document_date, created_at,
                       extracted_data, file_name, gstin
                FROM documents
                WHERE id IN ({placeholders}) AND user_id = ?
            """, (*doc_ids, user_id))

            rows = cursor.fetchall()
            doc_details = {row["id"]: dict(row) for row in rows}
            conn.close()

            enriched = []
            for doc in docs:
                doc_id = doc["doc_id"]
                enriched_doc = {
                    "doc_id": doc_id,
                    "similarity_score": doc.get("similarity_score", 0),
                    "text_preview": doc.get("text", "")[:500]
                }

                if doc_id in doc_details:
                    details = doc_details[doc_id]
                    enriched_doc.update({
                        "title": details.get("title") or details.get("file_name"),
                        "type": details.get("document_type"),
                        "amount": details.get("total_amount"),
                        "currency": details.get("currency"),
                        "vendor": details.get("vendor_name"),
                        "invoice_number": details.get("invoice_number"),
                        "date": details.get("document_date") or details.get("created_at"),
                        "gstin": details.get("gstin")
                    })

                    # Parse extracted_data JSON if available
                    if details.get("extracted_data"):
                        try:
                            extracted = json.loads(details["extracted_data"])
                            enriched_doc["extracted"] = extracted
                        except:
                            pass

                enriched.append(enriched_doc)

            return enriched

        except Exception as e:
            logger.error(f"Document enrichment failed: {e}")
            return docs  # Return unenriched docs on failure

    def _generate_answer(
        self,
        user_query: str,
        documents: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate answer using GPT-4o with function calling.
        Falls back to Anthropic if GPT-4o fails.
        """
        # Prepare documents context
        docs_context = self._format_documents_for_prompt(documents)

        system_prompt = """You are a document analysis assistant. Answer the user's question based on the provided documents.

INSTRUCTIONS:
1. Answer using only the information from the provided documents
2. Be concise but informative
3. Cite specific document details when possible
4. If information is missing, say so clearly
5. Format currency amounts clearly with their currency

If you cannot answer from the documents, indicate low confidence."""

        user_message = f"""User Question: {user_query}

Retrieved Documents:
{docs_context}

Provide a natural language answer based on these documents.
Use the answer_question function to provide your response."""

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
            functions=[ANSWER_QUESTION_FUNCTION],
            temperature=0.2,
            max_tokens=1500
        )

        if result.success and result.function_calls:
            # Extract function call result
            func_call = result.function_calls[0]
            try:
                if isinstance(func_call.get("arguments"), str):
                    args = json.loads(func_call["arguments"])
                else:
                    args = func_call["arguments"]

                return {
                    "answer": args.get("answer", ""),
                    "confidence": args.get("confidence", 0),
                    "citations": args.get("citations", []),
                    "provider_used": result.provider_used.value
                }
            except Exception as e:
                logger.error(f"Failed to parse answer function result: {e}")

        # Fallback: extract from content
        if result.success and result.data:
            content = result.data.get("content", "")
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(content)
                return {
                    "answer": parsed.get("answer", content),
                    "confidence": parsed.get("confidence", 0.7),
                    "citations": parsed.get("citations", []),
                    "provider_used": result.provider_used.value
                }
            except:
                # Return raw content as answer
                return {
                    "answer": content,
                    "confidence": 0.7,
                    "citations": [],
                    "provider_used": result.provider_used.value
                }

        # Both failed
        return {
            "answer": "I couldn't generate an answer based on the documents.",
            "confidence": 0,
            "citations": [],
            "provider_used": "failed",
            "error": result.error
        }

    def _format_documents_for_prompt(self, documents: List[Dict]) -> str:
        """Format documents for LLM context."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            parts = [f"Document {i} (ID: {doc['doc_id']})"]

            if doc.get("title"):
                parts.append(f"Title: {doc['title']}")
            if doc.get("type"):
                parts.append(f"Type: {doc['type']}")
            if doc.get("vendor"):
                parts.append(f"Vendor: {doc['vendor']}")
            if doc.get("amount"):
                currency = doc.get("currency", "INR")
                parts.append(f"Amount: {doc['amount']} {currency}")
            if doc.get("date"):
                parts.append(f"Date: {doc['date']}")
            if doc.get("similarity_score"):
                parts.append(f"Relevance: {doc['similarity_score']}%")

            if doc.get("extracted"):
                parts.append(f"Extracted Data: {json.dumps(doc['extracted'], indent=2)[:500]}")
            elif doc.get("text_preview"):
                parts.append(f"Content Preview: {doc['text_preview'][:300]}")

            formatted.append("\n".join(parts))

        return "\n\n---\n\n".join(formatted)

    def search_and_summarize(
        self,
        topic: str,
        user_id: int,
        db_path: str = "bot_data.db"
    ) -> Dict[str, Any]:
        """
        Search for documents about a topic and provide a summary.
        """
        result = self.answer_question(
            user_query=f"Tell me about {topic}",
            user_id=user_id,
            n_results=10,
            db_path=db_path
        )

        if not result["success"]:
            return result

        # Generate a focused summary
        docs = result.get("documents", [])
        if not docs:
            return {
                "success": False,
                "summary": f"I couldn't find any documents about '{topic}'.",
                "topic": topic
            }

        # Build summary from retrieved documents
        summary_parts = []
        total_docs = len(docs)

        summary_parts.append(f"Found {total_docs} documents related to '{topic}':\n")

        for i, doc in enumerate(docs[:5], 1):
            doc_info = []
            if doc.get("title"):
                doc_info.append(f"**{doc['title']}**")
            if doc.get("vendor"):
                doc_info.append(f"Vendor: {doc['vendor']}")
            if doc.get("amount"):
                currency = doc.get("currency", "INR")
                doc_info.append(f"Amount: {doc['amount']} {currency}")
            if doc.get("similarity_score"):
                doc_info.append(f"Match: {doc['similarity_score']}%")

            summary_parts.append(f"{i}. {' | '.join(doc_info)}")

        if total_docs > 5:
            summary_parts.append(f"\n_... and {total_docs - 5} more documents_")

        result["summary"] = "\n".join(summary_parts)
        return result


# Singleton instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create the RAG Service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
