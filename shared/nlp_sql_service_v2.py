"""
NLP to SQL Service v2 - Using GPT-4o with Function Calling via Orchestration Layer

This version uses the orchestrator to:
- Call GPT-4o as primary model
- Fallback to Anthropic if needed
- Use function calling for structured outputs
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import orchestration layer
from sqlalchemy import text

from shared.orchestrator import get_orchestrator, AgentType, ModelProvider, AgentStep
from shared.vector_service import get_vector_service
from shared.database import DatabaseService, engine

load_dotenv()

logger = logging.getLogger("nlp_sql_v2")


def _parse_extracted_json_column(raw: Any) -> Any:
    """Parse documents.extracted_data (JSON text). No field-specific logic."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def _rows_for_llm(filtered_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LLM context from DB only (user-scoped rows): columns + full extracted_data JSON."""
    rows: List[Dict[str, Any]] = []
    for item in filtered_data:
        kind = item.get("entry_kind") or item.get("type")
        if kind == "user_text_entry":
            rows.append({
                "entry_kind": "user_text_entry",
                "text_entry_id": item.get("text_entry_id"),
                "text": item.get("user_text_body"),
                "amount": item.get("amount"),
                "currency": item.get("currency"),
                "expense_category": item.get("expense_category"),
                "created_at": item.get("date"),
                "similarity_pct": item.get("_score"),
            })
            continue

        ext = item.get("extracted_data")
        parsed = _parse_extracted_json_column(ext)
        row: Dict[str, Any] = {
            "entry_kind": "document",
            "document_id": item.get("id"),
            "vendor_name": item.get("vendor"),
            "total_amount": item.get("amount"),
            "title": item.get("title"),
            "document_date": item.get("date"),
            "similarity_pct": item.get("_score"),
        }
        if parsed is not None:
            row["extracted_data"] = parsed
        elif isinstance(ext, str) and ext.strip():
            row["extracted_data_raw"] = ext[:20000]
        rows.append(row)
    return rows


# Database schema for context
DB_SCHEMA = """
Tables:

1. users
   - id (INTEGER PRIMARY KEY)
   - telegram_id (BIGINT UNIQUE)
   - first_name (VARCHAR)
   - last_name (VARCHAR)
   - username (VARCHAR)
   - created_at (DATETIME)
   - updated_at (DATETIME)

2. documents
   - id (INTEGER PRIMARY KEY)
   - user_id (INTEGER FOREIGN KEY -> users.id)
   - file_name (VARCHAR)
   - mime_type (VARCHAR)
   - file_size (INTEGER)
   - extracted_data (TEXT - JSON)
   - document_type (VARCHAR)
   - title (VARCHAR)
   - document_date (VARCHAR)
   - total_amount (FLOAT)
   - currency (VARCHAR)
   - vendor_name (VARCHAR)
   - invoice_number (VARCHAR)
   - gstin (VARCHAR)
   - confidence_overall (FLOAT)
   - raw_text (TEXT)
   - created_at (DATETIME)

3. user_text_entries
   - id (INTEGER PRIMARY KEY)
   - user_id (INTEGER FOREIGN KEY -> users.id)
   - text (TEXT)
   - intent_tag (VARCHAR)
   - expense_category (VARCHAR)
   - amount (FLOAT)
   - currency (VARCHAR)
   - created_at (DATETIME)

Key notes:
- Always filter by user_id for security
- extracted_data contains full JSON from OCR with ALL details including:
  * amounts: {total, subtotal, tax, currency}
  * vendor_or_sender: {name, address, contact}
  * items: array with {description, quantity, price, amount}
  * identifiers: {invoice_number, gstin, pan}
  * Example JSON: {"amounts": {"total": 1749, "subtotal": 1482.76, "tax": 266.24}, "vendor_or_sender": {"name": "Cloudtail India"}}
- Amazon documents may have vendor names like "Cloudtail India", "Appario Retail", etc. - use LIKE '%amazon%' OR check vendor_name for these sellers
- For tax rate queries: Calculate as (amounts.tax / amounts.subtotal) * 100 or use json_extract
- For item-level queries ("kya kya laya", "what items"), return extracted_data as raw_data field and let formatter summarize items
- document_type examples: invoice, receipt, product_listing, etc.
- Amounts are in the currency specified (mostly INR)
- user_text_entries stores manual user expense text (e.g. "Maine room rent 8000 diya").
  For expense totals/category queries, include this table along with documents when relevant.
"""

# Function schema for generate_sql function calling
GENERATE_SQL_FUNCTION = {
    "name": "generate_sql_query",
    "description": "Generate a safe SQL query based on the user's natural language request",
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQLite SQL query. Must include WHERE user_id filter."
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of what the query does"
            },
            "is_safe": {
                "type": "boolean",
                "description": "Whether the query is safe (has user_id filter, is SELECT only)"
            }
        },
        "required": ["sql", "explanation", "is_safe"]
    }
}

# Function schema for intent classification
CLASSIFY_INTENT_FUNCTION = {
    "name": "classify_intent",
    "description": "Classify the user's intent from their query",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["sql_query", "semantic_search", "conversation", "greeting", "user_info", "unknown"],
                "description": "The classified intent"
            },
            "needs": {
                "type": "string",
                "description": "Description of what the user wants"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1"
            },
            "suggested_action": {
                "type": "string",
                "enum": ["generate_sql", "vector_fallback", "chat_response", "greet", "explain_bot"],
                "description": "Suggested action based on intent"
            }
        },
        "required": ["intent", "needs", "confidence", "suggested_action"]
    }
}

# Function schema for deciding summary routing
CLASSIFY_SUMMARY_ROUTING_FUNCTION = {
    "name": "classify_summary_routing",
    "description": "Decide if query asks for global expense summary across all documents",
    "parameters": {
        "type": "object",
        "properties": {
            "use_global_summary_sql": {
                "type": "boolean",
                "description": "True when query requests overall totals/overview snapshot across all expenses"
            },
            "reason": {
                "type": "string",
                "description": "Short reason for the decision"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence between 0 and 1"
            }
        },
        "required": ["use_global_summary_sql", "reason", "confidence"]
    }
}

# Function schema for deciding whether plain user text should be stored
CLASSIFY_STORAGE_DECISION_FUNCTION = {
    "name": "classify_storage_decision",
    "description": "Decide whether a plain user text should be saved as expense-related entry, assign exact category, and infer emotional tone",
    "parameters": {
        "type": "object",
        "properties": {
            "should_store": {
                "type": "boolean",
                "description": "True when message is a real expense/payment statement that should be stored"
            },
            "category": {
                "type": "string",
                "description": "Must be exactly one string from the expense category list provided in the system prompt (use 'Other' if unclear)"
            },
            "user_emotion": {
                "type": "string",
                "enum": ["neutral", "positive", "negative", "stressed_or_urgent", "grateful", "casual"],
                "description": "Overall emotional tone of the message (for reply styling), independent of should_store"
            },
            "reason": {
                "type": "string",
                "description": "Short explanation of decision"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence between 0 and 1"
            }
        },
        "required": ["should_store", "category", "user_emotion", "reason", "confidence"]
    }
}


class NLPSQLServiceV2:
    """
    NLP to SQL Service using GPT-4o with Function Calling via Orchestration Layer.

    Architecture:
    1. Intent Classification (GPT-4o with function calling)
    2. SQL Generation (GPT-4o with function calling) -> Anthropic fallback
    3. Query Execution (SQLite)
    4. Response Formatting (GPT-4o)
    """

    def __init__(self):
        # Get orchestrator (control center)
        self.orchestrator = get_orchestrator()
        # Vector service for fallback
        self.vector_service = get_vector_service()
        self.expense_categories = DatabaseService.EXPENSE_CATEGORIES
        # Conversation history: {user_id: [{"query": str, "response": str}, ...]}
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        self.MAX_HISTORY = 10

        logger.info("NLPSQLServiceV2 initialized with orchestration layer")

    def _get_conversation_context(self, user_id: int) -> str:
        """Get formatted conversation history for the user."""
        if user_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[user_id]
        if not history:
            return ""
        
        context_lines = []
        for h in history:
            try:
                if isinstance(h, dict):
                    query = h.get('query', '')
                    response = h.get('response', '')
                else:
                    # Fallback if stored as string (shouldn't happen)
                    continue
                if query and response:
                    context_lines.append(f"User: {query}")
                    context_lines.append(f"Assistant: {response}")
            except Exception:
                continue
        
        if not context_lines:
            return ""
        
        context = "\n".join(context_lines)
        return f"\n\n=== PREVIOUS CONVERSATION HISTORY (CRITICAL FOR PRONOUN RESOLUTION) ===\n{context}\n=== END OF HISTORY ===\n"

    def _add_to_history(self, user_id: int, query: str, response: str):
        """Add interaction to conversation history, keeping only last 10."""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            "query": query,
            "response": response
        })
        
        logger.info(f"Added to history for user {user_id}: query='{query[:50]}...', total entries={len(self.conversation_history[user_id])}")
        
        # Keep only last 10
        if len(self.conversation_history[user_id]) > self.MAX_HISTORY:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.MAX_HISTORY]

    def clear_user_history(self, user_id: int) -> None:
        """Clear conversation history for a specific user."""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
            logger.info(f"Cleared NLP SQL conversation history for user {user_id}")

    def _is_global_expense_summary_query(self, user_query: str) -> bool:
        """Detect broad summary/overview asks that should always use aggregate SQL."""
        q = (user_query or "").strip().lower()
        if not q:
            return False

        summary_terms = [
            "summary", "overview", "snapshot", "total expenses", "expense summary",
            "expenses ka summary", "summary dedo", "summarize", "overall"
        ]
        hindi_summary_terms = [
            "kharcha", "kharche", "expense", "expenses", "total", "saare", "sabhi"
        ]
        has_summary_intent = any(term in q for term in summary_terms)
        has_expense_hint = any(term in q for term in hindi_summary_terms)

        # Also catch short asks like: "summary dedo expenses ka"
        regex_match = re.search(r"(summary|overview|snapshot).*(expense|expenses|kharch|kharc|total)", q)
        return bool((has_summary_intent and has_expense_hint) or regex_match)

    def _build_global_summary_sql(self, user_id: int) -> str:
        """Return canonical aggregate SQL for global expense summary requests."""
        return f"""SELECT
  (SELECT COUNT(*) FROM documents WHERE user_id = {user_id}) + (SELECT COUNT(*) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL) as total_documents,
  COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id}), 0) + COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL), 0) as total_amount,
  (SELECT COUNT(DISTINCT vendor_name) FROM documents WHERE user_id = {user_id} AND vendor_name IS NOT NULL) as unique_vendors,
  (
    COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id}), 0) + COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL), 0)
  ) / NULLIF(
    (SELECT COUNT(*) FROM documents WHERE user_id = {user_id}) + (SELECT COUNT(*) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL),
    0
  ) as avg_amount"""

    def _is_rent_query(self, user_query: str) -> bool:
        q = (user_query or "").lower()
        return any(k in q for k in ["rent", "room rent", "house rent", "kiraya"])

    def _build_rent_total_sql(self, user_id: int) -> str:
        """Rent total across OCR documents + manual user text entries."""
        return f"""SELECT
  COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id} AND (LOWER(title) LIKE '%rent%' OR LOWER(vendor_name) LIKE '%rent%' OR LOWER(extracted_data) LIKE '%rent%')), 0) as document_rent_total,
  COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL AND (LOWER(expense_category) = 'rent' OR LOWER(text) LIKE '%rent%' OR LOWER(text) LIKE '%kiraya%')), 0) as text_rent_total,
  (
    COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id} AND (LOWER(title) LIKE '%rent%' OR LOWER(vendor_name) LIKE '%rent%' OR LOWER(extracted_data) LIKE '%rent%')), 0)
    +
    COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL AND (LOWER(expense_category) = 'rent' OR LOWER(text) LIKE '%rent%' OR LOWER(text) LIKE '%kiraya%')), 0)
  ) as total_rent"""

    def _should_use_global_summary_sql(self, user_query: str, user_id: int) -> bool:
        """
        Dynamically decide summary routing via prompt+function-calling.
        Falls back to keyword heuristic if model parsing fails.
        """
        context = self._get_conversation_context(user_id)
        system_prompt = """You classify whether a user query should be answered using a GLOBAL aggregate expense summary.

Return use_global_summary_sql=true only when user asks for overall totals/snapshot/overview across all (or many) expenses/documents.

Set use_global_summary_sql=false when:
- user asks about one specific vendor/entity (e.g., "maatha ka summary")
- user asks item-level/details/doc listing
- user asks a narrow follow-up referring to one bill/document

Examples:
- "summary dedo expenses ka" -> true
- "overall expense overview" -> true
- "maatha agencies ka summary do" -> false
- "amazon ka bill kitna tha" -> false
- "kya kya items laya tha" -> false
"""
        user_message = f"""Query: "{user_query}"
{context}
Classify routing now."""

        try:
            result = self.orchestrator.execute_with_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=[CLASSIFY_SUMMARY_ROUTING_FUNCTION],
                temperature=0.1,
                max_tokens=300
            )
            if result.success and result.function_calls:
                func_call = result.function_calls[0]
                args = func_call["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                decision = bool(args.get("use_global_summary_sql", False))
                confidence = float(args.get("confidence", 0.0) or 0.0)
                if confidence >= 0.60:
                    return decision
        except Exception as e:
            logger.warning(f"Summary routing classifier failed; using heuristic fallback: {e}")

        # Fallback to local heuristic
        return self._is_global_expense_summary_query(user_query)

    def _understand_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Step 1: Understand user intent using GPT-4o with function calling.
        """
        system_prompt = """You are an intent classifier for a document management bot.

Analyze the user's query and classify the intent.

Intent Types:
- "sql_query": User wants specific data from their documents (show invoices, find receipts, totals, etc.)
- "semantic_search": User asks about content/topics in documents (tell me about AI, what do I have about X, etc.)
- "conversation": General chat, questions about the bot, help, etc.
- "greeting": Hello, hi, thanks, bye, etc.
- "user_info": User asks about their profile, name, settings
- "unknown": Unclear what user wants

Examples:
User: "Show my invoices" → sql_query, needs: list of invoice documents, confidence: 0.95
User: "Tell me about AI" → semantic_search, needs: find documents related to AI topic, confidence: 0.9
User: "Give me summary of akash enterprises" → semantic_search, needs: find documents about akash enterprises entity, confidence: 0.9
User: "Total amount across all my receipts" → sql_query, needs: sum of amounts for receipt documents, confidence: 0.9
User: "Hello" → greeting, needs: greeting response, confidence: 0.99

Use the classify_intent function to provide your classification."""

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=f"Classify this query: {user_query}",
            functions=[CLASSIFY_INTENT_FUNCTION],
            temperature=0.1,
            max_tokens=1024
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
                    "intent": args.get("intent", "unknown"),
                    "needs": args.get("needs", "could not understand"),
                    "confidence": args.get("confidence", 0.0),
                    "suggested_action": args.get("suggested_action", "vector_fallback"),
                    "provider_used": result.provider_used.value
                }
            except Exception as e:
                logger.error(f"Failed to parse intent function result: {e}")

        # Fallback: try to parse from content
        if result.success and result.data:
            content = result.data.get("content", "")
            try:
                # Try to extract JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(content)
                return {
                    "intent": parsed.get("intent", "unknown"),
                    "needs": parsed.get("needs", "could not understand"),
                    "confidence": parsed.get("confidence", 0.0),
                    "suggested_action": parsed.get("suggested_action", "vector_fallback"),
                    "provider_used": result.provider_used.value
                }
            except Exception as e:
                logger.error(f"Failed to parse intent content: {e}")

        # Default fallback
        return {
            "intent": "unknown",
            "needs": "could not understand query",
            "confidence": 0.0,
            "suggested_action": "vector_fallback",
            "provider_used": "fallback"
        }

    def _normalize_expense_category_label(self, raw: Optional[str]) -> str:
        """Map model output to a canonical category from DatabaseService.EXPENSE_CATEGORIES."""
        return DatabaseService.normalize_expense_category_label(raw)

    def classify_plain_text_expense(self, user_text: str) -> Dict[str, Any]:
        """
        Classify a plain Telegram message: expense vs non-expense, exact category (if expense),
        emotional tone, and confidence. Used for DB + vector persistence and reply styling.
        """
        text = (user_text or "").strip()
        empty = {
            "should_store": False,
            "category": "Other",
            "user_emotion": "neutral",
            "confidence": 0.0,
        }
        if not text:
            return empty
        logger.info("Storage decision: evaluating user text intent: '%s'", text)

        category_lines = "\n".join(f"- {c}" for c in self.expense_categories)
        system_prompt = f"""You classify a user's plain chat message for a receipt/expense Telegram bot.

ALLOWED expense categories (category MUST be exactly one of these strings, character-for-character):
{category_lines}

Tasks:
1) should_store: true only if the user is logging or stating money they spent or paid (expense, bill paid, rent paid, purchase, etc.). false for greetings, thanks, bot help, questions about past data, jokes, or unclear intent.
2) category: If should_store is true, pick the single best category from the list above. If should_store is false, still output category=Other.
3) user_emotion: How the user sounds emotionally (for reply tone) — neutral, positive, negative, stressed_or_urgent, grateful, or casual.
4) confidence: 0–1 for your should_store decision.
5) If unsure about storing, prefer should_store=false.
6) Reply only via the classify_storage_decision function call."""

        user_message = f"""User message:
\"\"\"{text}\"\"\"

Classify now."""

        try:
            result = self.orchestrator.execute_with_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=[CLASSIFY_STORAGE_DECISION_FUNCTION],
                temperature=0.1,
                max_tokens=400
            )
            if result.success and result.function_calls:
                func_call = result.function_calls[0]
                args = func_call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                should_store_raw = bool(args.get("should_store", False))
                confidence = float(args.get("confidence", 0.0) or 0.0)
                category_norm = self._normalize_expense_category_label(args.get("category"))
                emotion = args.get("user_emotion") or "neutral"
                if emotion not in (
                    "neutral",
                    "positive",
                    "negative",
                    "stressed_or_urgent",
                    "grateful",
                    "casual",
                ):
                    emotion = "neutral"
                final_store = should_store_raw and confidence >= 0.55
                logger.info(
                    "Storage decision result: should_store=%s confidence=%.2f category=%s emotion=%s",
                    final_store,
                    confidence,
                    category_norm if final_store else "Other",
                    emotion,
                )
                return {
                    "should_store": final_store,
                    "category": category_norm if final_store else "Other",
                    "user_emotion": emotion,
                    "confidence": confidence,
                }
        except Exception:
            logger.exception("Expense-text storage decision failed")

        return empty

    def should_store_as_expense_text(self, user_text: str) -> bool:
        """Backward-compatible: true if plain text should be persisted as an expense entry."""
        return bool(self.classify_plain_text_expense(user_text).get("should_store"))

    def generate_sql(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """
        Step 2: Generate SQL using GPT-4o with function calling.
        Falls back to Anthropic if GPT-4o fails.
        """
        system_prompt = f"""You are an AI assistant that converts natural language questions into safe SQLite SQL queries.

DATABASE SCHEMA:
{DB_SCHEMA}

========================
IMPORTANT: CONVERSATION CONTEXT
========================
The user may ask follow-up questions with pronouns like "us bill me" (in that bill), "uska GST" (its GST), "kitna tha" (how much was).
Use the conversation context to resolve these pronouns to specific vendors/entities from previous queries.
Example:
- Context: "User: Amazon ka kitna bill tha / Bot: Amazon ka bill ₹1,749 hai"
- New query: "Gst tha us bill me" → Should query for Amazon documents, not random vendors

========================
IMPORTANT: AMAZON SELLER NAMES
========================
Amazon documents may have vendor names like "Cloudtail India", "Appario Retail", "Cloudtail India Private Limited", etc.
When user asks about "Amazon", search for these seller names using LIKE with OR conditions.
Example: vendor_name LIKE '%amazon%' OR vendor_name LIKE '%cloudtail%' OR vendor_name LIKE '%appario%'

========================
CRITICAL SECURITY RULES:
========================
1. ALWAYS include "WHERE user_id = {user_id}" in the query (MANDATORY)
2. NEVER access data of other users
3. ONLY generate SELECT queries (NO INSERT, UPDATE, DELETE, DROP, ALTER)
4. "my" always refers to user_id = {user_id}
5. Return only valid SQLite SQL

========================
UI RESPONSE FORMAT RULES (VERY IMPORTANT):
========================
The frontend expects these columns ONLY (NO id, NO user_id):

- type (document_type AS type)
- title
- amount (total_amount AS amount)
- vendor (vendor_name AS vendor)
- date (created_at AS date)

👉 NEVER use SELECT *
👉 NEVER include id or user_id in SELECT
👉 ALWAYS explicitly select columns above only
👉 ALWAYS use aliases exactly as above

Exception:
- For aggregate/summary queries (total/count/overview/snapshot), return aggregate columns
  that best answer the question instead of the list columns above.

========================
 COLUMN MAPPING:
========================
- document_type → type
- total_amount → amount
- vendor_name → vendor
- created_at → date

========================
 SPECIAL CASES:
========================

1. If user asks about vendors (e.g., "Who are my vendors?")
   - Use vendor_name as title
   - Set type = 'vendor'
   - Set amount = NULL

2. If query is about totals (SUM, COUNT, etc.)
   - Return aggregated values with clear aliases
   - For full expense summary/overview queries (include manual text expenses too), use:
     SELECT
       (SELECT COUNT(*) FROM documents WHERE user_id = {user_id}) + (SELECT COUNT(*) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL) as total_documents,
       COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id}), 0) + COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL), 0) as total_amount,
       COUNT(DISTINCT vendor_name) as unique_vendors,
       (
         COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id}), 0) + COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL), 0)
       ) / NULLIF(
         (SELECT COUNT(*) FROM documents WHERE user_id = {user_id}) + (SELECT COUNT(*) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL),
         0
       ) as avg_amount
     FROM documents
     WHERE user_id = {user_id}
   - Do NOT return a single invoice row for summary/overview asks.
   - If user asks "summary", "overview", "snapshot", "expenses ka summary",
     "summary dedo", or "total expenses", ALWAYS return exactly 1 aggregate row.
   - Never use ORDER BY + LIMIT 1 for summary/overview requests.
   - Never answer summary/overview requests with vendor-specific single bill queries.

3. If user asks about their profile/details:
   - Query the users table
   - Example: SELECT first_name, last_name, username FROM users WHERE id = {user_id}

4. If user asks about ITEMS/PRODUCTS/LINE ITEMS (e.g., "kya kya item laya", "what items", "kaunse products"):
   - Return the raw extracted_data JSON field - let the response formatter summarize items from it
   - Use: SELECT extracted_data as raw_data, vendor_name as vendor, title FROM documents WHERE user_id = {user_id} AND vendor_name LIKE '%maatha%'
   - The raw_data contains an "items" array with objects having: description, quantity, price, amount
   - Do NOT use json_extract - return the full extracted_data field

5. If user asks about TAX RATE/GST PERCENTAGE (e.g., "kitna % tax h", "GST percentage"):
   - Extract tax and subtotal from amounts field
   - Use: json_extract(extracted_data, '$.amounts.tax') as tax, json_extract(extracted_data, '$.amounts.subtotal') as subtotal
   - Example query for Amazon: SELECT json_extract(extracted_data, '$.amounts.tax') as tax, json_extract(extracted_data, '$.amounts.subtotal') as subtotal, vendor_name as vendor FROM documents WHERE user_id = {user_id} AND (vendor_name LIKE '%amazon%' OR vendor_name LIKE '%cloudtail%' OR vendor_name LIKE '%appario%')
   - Return: tax, subtotal, vendor

6. If user asks rent-related totals ("room rent kitna", "rent total"):
   - Include BOTH documents + user_text_entries amounts.
   - Example:
     SELECT
       COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id} AND (LOWER(title) LIKE '%rent%' OR LOWER(vendor_name) LIKE '%rent%' OR LOWER(extracted_data) LIKE '%rent%')), 0) as document_rent_total,
       COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL AND (LOWER(expense_category) = 'rent' OR LOWER(text) LIKE '%rent%' OR LOWER(text) LIKE '%kiraya%')), 0) as text_rent_total,
       (
         COALESCE((SELECT SUM(total_amount) FROM documents WHERE user_id = {user_id} AND (LOWER(title) LIKE '%rent%' OR LOWER(vendor_name) LIKE '%rent%' OR LOWER(extracted_data) LIKE '%rent%')), 0)
         +
         COALESCE((SELECT SUM(amount) FROM user_text_entries WHERE user_id = {user_id} AND amount IS NOT NULL AND (LOWER(expense_category) = 'rent' OR LOWER(text) LIKE '%rent%' OR LOWER(text) LIKE '%kiraya%')), 0)
       ) as total_rent

Use the generate_sql_query function to provide your SQL."""

        context = self._get_conversation_context(user_id)
        user_message = f"Convert this query to SQL: {user_query}{context}"
        logger.info(f"Query: {user_query}")
        logger.info(f"Context length: {len(context)} chars")
        logger.info(f"Context preview: {context[:500] if context else 'None'}")

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
            functions=[GENERATE_SQL_FUNCTION],
            temperature=0.1,
            max_tokens=2048,
            max_retries=2
        )

        if result.success and result.function_calls:
            # Extract function call result
            func_call = result.function_calls[0]
            try:
                if isinstance(func_call.get("arguments"), str):
                    args = json.loads(func_call["arguments"])
                else:
                    args = func_call["arguments"]

                sql = args.get("sql", "")
                is_safe = args.get("is_safe", False)

                # Additional safety checks
                if sql:
                    has_user_filter = f"user_id = {user_id}" in sql or f"id = {user_id}" in sql
                    is_select = sql.strip().upper().startswith("SELECT")

                    if not has_user_filter or not is_select:
                        return {
                            "sql": None,
                            "explanation": None,
                            "is_safe": False,
                            "error": f"Security error: Query must filter by user_id = {user_id} and be SELECT only",
                            "provider_used": result.provider_used.value
                        }

                return {
                    "sql": sql,
                    "explanation": args.get("explanation", ""),
                    "is_safe": is_safe,
                    "error": None,
                    "provider_used": result.provider_used.value
                }
            except Exception as e:
                logger.error(f"Failed to parse SQL function result: {e}")

        # Fallback: try to parse from content (for Anthropic fallback)
        if result.success and result.data:
            content = result.data.get("content", "")
            try:
                # Try to extract JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(content)

                sql = parsed.get("sql", "")
                if sql:
                    has_user_filter = f"user_id = {user_id}" in sql or f"id = {user_id}" in sql
                    is_select = sql.strip().upper().startswith("SELECT")

                    if not has_user_filter or not is_select:
                        return {
                            "sql": None,
                            "explanation": None,
                            "is_safe": False,
                            "error": f"Security error: Query must filter by user_id = {user_id}",
                            "provider_used": result.provider_used.value
                        }

                return {
                    "sql": sql,
                    "explanation": parsed.get("explanation", ""),
                    "is_safe": parsed.get("is_safe", False),
                    "error": parsed.get("error"),
                    "provider_used": result.provider_used.value
                }
            except Exception as e:
                logger.error(f"Failed to parse SQL content: {e}")

        # Both models failed
        return {
            "sql": None,
            "explanation": None,
            "is_safe": False,
            "error": result.error or "Failed to generate SQL",
            "provider_used": "failed"
        }

    def execute_query(self, sql: str, db_path: str = None) -> Dict[str, Any]:
        """Execute SQL against the configured app database (SQLite or PostgreSQL via DATABASE_URL)."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                if result.returns_rows:
                    rows_raw = result.mappings().all()
                    result_rows = [dict(row) for row in rows_raw]
                    columns = list(result.keys())
                else:
                    result_rows = []
                    columns = []

                return {
                    "columns": columns,
                    "rows": result_rows,
                    "row_count": len(result_rows),
                    "error": None,
                }

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": str(e),
            }

    def _generate_social_response(self, user_query: str, mode: str = "conversation") -> str:
        """Generate greeting/conversation replies in the same language as user's message."""
        if mode == "greeting":
            intent_hint = "The user greeted you."
            default_reply = "Hi! Main help ke liye yahin hoon 😊 Aap documents ya expenses ke bare me pooch sakte ho."
        elif mode == "user_info":
            intent_hint = "The user asked profile/help capabilities."
            default_reply = "Aap profile details, document stats, vendors, total amount ya GST summary puch sakte ho 😊"
        else:
            intent_hint = "The user is having a general conversation."
            default_reply = "Bilkul! Aap natural language me pucho, main documents se sahi answer nikal dunga 😊"

        system_prompt = """You are a friendly document and expense assistant.

Rules:
1. Reply in the SAME language/script as the user's latest message.
2. Keep tone warm, natural, and concise.
3. Add 1 relevant emoji only when it feels natural.
4. Do not mention SQL, database, models, or technical internals.
5. If user seems unsure, include one helpful example query."""

        user_message = f"""{intent_hint}
User message: {user_query}
Generate one short assistant reply."""

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
            functions=[],
            temperature=0.6,
            max_tokens=220
        )

        if result.success and result.data and result.data.get("content"):
            return result.data["content"].strip()
        return default_reply

    def ask_ai(self, user_query: str, user_id: int, db_path: str = "bot_data.db") -> Dict[str, Any]:
        """
        Complete pipeline with orchestration:
        1. Understand Intent (GPT-4o)
        2. Route to appropriate handler
        3. Generate SQL (GPT-4o) or fallback to Vector Search
        4. Execute Query
        5. Format Response (GPT-4o)
        """
        # Guardrail: deterministic handling for full expense summary asks
        if self._should_use_global_summary_sql(user_query, user_id):
            sql = self._build_global_summary_sql(user_id)
            exec_result = self.execute_query(sql, db_path)

            if exec_result.get("error"):
                logger.info("Deterministic summary SQL failed, falling back to intent pipeline")
            else:
                data = exec_result.get("rows", [])
                ai_response = self._format_response(user_query, sql, data, user_id)
                self._add_to_history(user_id, user_query, ai_response)
                return {
                    "success": True,
                    "sql": sql,
                    "explanation": "Deterministic aggregate summary query",
                    "error": None,
                    "data": data,
                    "row_count": exec_result.get("row_count", 0),
                    "columns": exec_result.get("columns", []),
                    "ai_response": ai_response,
                    "provider_used": "deterministic"
                }

        # Guardrail: deterministic rent total including manual text entries.
        if self._is_rent_query(user_query):
            sql = self._build_rent_total_sql(user_id)
            exec_result = self.execute_query(sql, db_path)
            if not exec_result.get("error"):
                data = exec_result.get("rows", [])
                ai_response = self._format_response(user_query, sql, data, user_id)
                self._add_to_history(user_id, user_query, ai_response)
                return {
                    "success": True,
                    "sql": sql,
                    "explanation": "Deterministic rent total query across documents + user_text_entries",
                    "error": None,
                    "data": data,
                    "row_count": exec_result.get("row_count", 0),
                    "columns": exec_result.get("columns", []),
                    "ai_response": ai_response,
                    "provider_used": "deterministic"
                }

        # Step 1: Intent Classification
        intent_analysis = self._understand_intent(user_query)
        logger.info(f"Intent analysis for '{user_query}': {intent_analysis}")

        intent = intent_analysis.get("intent", "unknown")
        confidence = intent_analysis.get("confidence", 0)

        # Route based on intent
        if intent == "semantic_search" or intent == "unknown" or confidence < 0.6:
            logger.info(f"Routing to semantic search (intent: {intent}, confidence: {confidence})")
            return self._vector_search_fallback(user_query, user_id, f"Intent: {intent}")

        if intent == "greeting":
            response = self._generate_social_response(user_query, mode="greeting")
            self._add_to_history(user_id, user_query, response)
            return {
                "success": True,
                "sql": None,
                "explanation": None,
                "error": None,
                "data": None,
                "row_count": 0,
                "columns": None,
                "ai_response": response
            }

        if intent == "conversation":
            response = self._generate_social_response(user_query, mode="conversation")
            self._add_to_history(user_id, user_query, response)
            return {
                "success": True,
                "sql": None,
                "explanation": None,
                "error": None,
                "data": None,
                "row_count": 0,
                "columns": None,
                "ai_response": response
            }

        if intent == "user_info":
            # Check if user wants to see vendors
            if "vendor" in user_query.lower() or "seller" in user_query.lower():
                vendors = DatabaseService.get_distinct_vendor_names(user_id)

                if vendors:
                    response = f"Aapke documents me ye vendors hain: {', '.join(vendors)}"
                else:
                    response = "Aapke documents me koi vendor information nahi hai."
                self._add_to_history(user_id, user_query, response)
                return {
                    "success": True,
                    "sql": None,
                    "explanation": None,
                    "error": None,
                    "data": None,
                    "row_count": 0,
                    "columns": None,
                    "ai_response": response
                }
            
            response = self._generate_social_response(user_query, mode="user_info")
            self._add_to_history(user_id, user_query, response)
            return {
                "success": True,
                "sql": None,
                "explanation": None,
                "error": None,
                "data": None,
                "row_count": 0,
                "columns": None,
                "ai_response": response
            }

        # Step 2: Generate SQL (via orchestrator with GPT-4o/Anthropic)
        sql_result = self.generate_sql(user_query, user_id)
        logger.info(f"Generated SQL: {sql_result.get('sql', 'NONE')}")

        # FALLBACK: If SQL generation fails
        if not sql_result.get("is_safe") or not sql_result.get("sql"):
            logger.info(f"SQL generation failed for user {user_id}, falling back to vector search")
            return self._vector_search_fallback(user_query, user_id, sql_result.get("error"))

        sql = sql_result["sql"]

        # Step 3: Execute query
        exec_result = self.execute_query(sql, db_path)

        # FALLBACK: If SQL execution fails or returns empty
        if exec_result.get("error"):
            logger.info(f"SQL execution failed for user {user_id}, falling back to vector search")
            return self._vector_search_fallback(user_query, user_id, exec_result["error"])

        if exec_result.get("row_count", 0) == 0:
            logger.info(f"SQL returned 0 rows for user {user_id}, falling back to vector search with context")
            return self._vector_search_fallback(user_query, user_id, "SQL query returned no matching rows")

        # Step 4: Format response
        data = exec_result["rows"]
        ai_response = self._format_response(user_query, sql, data, user_id)
        
        # Add to conversation history
        self._add_to_history(user_id, user_query, ai_response)

        # Step 5: Store SQL results in vector DB
        try:
            self._store_sql_results(user_query, sql, data, user_id)
        except Exception as e:
            logger.warning(f"Failed to store SQL results (non-critical): {e}")

        return {
            "success": True,
            "sql": sql,
            "explanation": sql_result.get("explanation"),
            "error": None,
            "data": data,
            "row_count": exec_result["row_count"],
            "columns": exec_result["columns"],
            "ai_response": ai_response,
            "provider_used": sql_result.get("provider_used", "unknown")
        }

    def _format_response(self, user_query: str, sql: str, data: List[Dict], user_id: int) -> str:
        """Format SQL results using GPT-4o via orchestrator."""
        # Deterministic reply for aggregate summary rows to avoid context bleed.
        if data and len(data) == 1 and {"total_documents", "total_amount", "unique_vendors", "avg_amount"}.issubset(set(data[0].keys())):
            row = data[0]
            total_docs = int(row.get("total_documents") or 0)
            total_amount = float(row.get("total_amount") or 0)
            unique_vendors = int(row.get("unique_vendors") or 0)
            avg_amount = float(row.get("avg_amount") or 0)
            return (
                f"Aapke {total_docs} expenses ka total ₹{total_amount:,.2f} hai. "
                f"Average ₹{avg_amount:,.2f} per expense hai, aur {unique_vendors} unique vendors hain."
            )
        if data and len(data) == 1 and {"document_rent_total", "text_rent_total", "total_rent"}.issubset(set(data[0].keys())):
            row = data[0]
            doc_rent = float(row.get("document_rent_total") or 0)
            txt_rent = float(row.get("text_rent_total") or 0)
            total_rent = float(row.get("total_rent") or 0)
            return (
                f"Aapka total room rent ₹{total_rent:,.2f} hai.\n"
                f"• Documents se: ₹{doc_rent:,.2f}\n"
                f"• Aapke text entries se: ₹{txt_rent:,.2f}"
            )

        q = (user_query or "").lower()
        is_list_request = any(token in q for token in ["list", "all", "saare", "sabhi", "vendors", "vendor", "sellers"])
        formatter_limit = 100 if is_list_request else 20
        formatter_rows = data[:formatter_limit]
        data_summary = json.dumps(formatter_rows, indent=2) if data else "[]"
        row_count = len(data)

        system_prompt = """You are a friendly expense/document assistant. Answer the user's question based on the query results.

Rules:
1. Answer ONLY what was asked - no extra information
2. Reply in the SAME language/script as the user's latest message.
3. Use natural, friendly conversational tone.
3. Format amounts/currency clearly with ₹ symbol
4. Do not mention SQL, database, or technical details
5. Keep it brief but complete - 2-5 short lines are okay when summary is needed
6. If user explicitly asks to list/show all vendors/items/rows, include all relevant rows from provided data.
   Do not silently reduce to top 5.
7. IMPORTANT: If data contains "raw_data" field with JSON, parse it to extract items, amounts, vendor details etc.
   - raw_data.items[] has objects with: description, quantity, price, amount
   - Summarize items as: "Item Name (qty)" format
   - Example: [{"description":"Cement","quantity":5}] → "Cement (5 qty)"
8. For totals/tax/GST questions, use clear friendly breakdown with bullets when multiple parts exist.
9. Use at most one relevant emoji where useful. Avoid over-decorating.

Examples:
User: "maatha agencies ka kitna bill h"
Data: [{"vendor_name": "MAATHA AGENCIES", "total_amount": 7222.40}]
→ "MAATHA AGENCIES ka bill ₹7,222.40 hai"

User: "show my invoices"
Data: [{"document_type": "invoice", "vendor_name": "ABC Corp"}]
→ "Aapke paas ABC Corp se invoice hai"

User: "kya kya item laya tha maatha agencies se"
Data: [{"raw_data": "{\"items\":[{\"description\":\"2x2 Single M. C. Ply\",\"quantity\":266,\"price\":1.75,\"amount\":113}]}", "vendor": "MAATHA AGENCIES"}]
→ "Maatha Agencies se 2x2 Single M. C. Ply (266 qty) laya tha"

User: "kya kya item laya tha" (generic, no vendor specified)
Data: [{"raw_data": "{\"items\":[{\"description\":\"Cement 50kg\",\"quantity\":5,\"price\":350,\"amount\":1750}]}", "vendor": "MAATHA"}, {"raw_data": "{\"items\":[{\"description\":\"Steel rods\",\"quantity\":10,\"price\":1200,\"amount\":12000}]}", "vendor": "ABC Corp"}]
→ "Cement 50kg (5 qty - Maatha Agencies) aur Steel rods (10 qty - ABC Corp) laya tha"

User: "kitna % tax h"
Data: [{"tax": "266.24", "subtotal": "1482.76", "vendor": "Cloudtail India"}]
→ "Tax rate 18% hai (₹266.24 tax on ₹1482.76 subtotal)."

User: "ab total GST kitna tha"
Data: [{"vendor":"Laptop","tax":4800},{"vendor":"Dana Pani","tax":1441.44}]
→ "Aapka total GST ₹6,241.44 hai.\n• Laptop: ₹4,800\n• Dana Pani: ₹1,441.44"

User: "summary dedo expenses ka"
Data: [{"total_documents":4,"total_amount":13695.64,"unique_vendors":4,"avg_amount":3423.91}]
→ "Aapke 4 expenses ka total ₹13,695.64 hai. Average ₹3,423.91 per expense hai, aur 4 unique vendors hain." """

        context = self._get_conversation_context(user_id)
        user_message = f"""Question: "{user_query}"

Data found ({row_count} rows):
{data_summary}

Answer directly in a conversational way.{context}"""

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
            functions=[],  # No function calling needed for formatting
            temperature=0.7,
            max_tokens=1024
        )

        if result.success and result.data:
            return result.data.get("content", "No summary available")

        # Fallback: return raw data summary
        if not data:
            return "No results found."
        lines = []
        for i, row in enumerate(formatter_rows, 1):
            row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
            lines.append(f"{i}. {row_text}")
        if len(data) > len(formatter_rows):
            lines.append(f"...and {len(data) - len(formatter_rows)} more rows")
        return "\n".join(lines)

    def _store_sql_results(self, user_query: str, sql: str, data: List[Dict], user_id: int) -> None:
        """Store SQL results in vector database for future semantic search."""
        if not data:
            return

        try:
            result_texts = [
                f"Query: {user_query}",
                f"SQL: {sql}",
                f"Results ({len(data)} rows):"
            ]

            for i, row in enumerate(data[:5]):
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                result_texts.append(f"  Row {i+1}: {row_text}")

            searchable_text = "\n".join(result_texts)

            import hashlib
            import time
            query_hash = hashlib.md5(f"{user_id}:{user_query}:{sql}".encode()).hexdigest()[:12]
            result_id = f"sql_result_{user_id}_{query_hash}_{int(time.time())}"

            # Use OpenAI for embeddings (separate from orchestrator LLM)
            import openai
            openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=searchable_text[:8000]
            )
            embedding = response.data[0].embedding

            self.vector_service.collection.upsert(
                ids=[result_id],
                embeddings=[embedding],
                metadatas=[{
                    "user_id": user_id,
                    "type": "sql_result",
                    "original_query": user_query,
                    "sql": sql[:500],
                    "row_count": len(data),
                    "text": searchable_text[:1000]
                }],
                documents=[searchable_text]
            )

            logger.info(f"Stored SQL results in vector DB: {result_id} for user {user_id}")

        except Exception as e:
            logger.error(f"Error storing SQL results: {e}")
            raise

    def _vector_search_fallback(self, user_query: str, user_id: int, error_reason: str = None) -> Dict[str, Any]:
        """Semantic search fallback when SQL fails."""
        import hashlib
        import time

        cache_key = f"{user_id}:{hashlib.md5(user_query.lower().encode()).hexdigest()[:16]}"
        _CACHE_TTL_SECONDS = 300

        if hasattr(self, '_search_cache') and cache_key in self._search_cache:
            cached_time, cached_result = self._search_cache[cache_key]
            if time.time() - cached_time < _CACHE_TTL_SECONDS:
                cached_result['cached'] = True
                return cached_result

        try:
            # Get conversation context to improve search
            context = self._get_conversation_context(user_id)
            enhanced_query = f"{user_query}{context}"
            
            results = self.vector_service.search(
                query=enhanced_query,
                user_id=user_id,
                n_results=10
            )
            logger.info(f"Vector search with context: query_len={len(user_query)}, context_len={len(context)}")

            # Supplement embedding hits with SQL substring match on vendor / extracted_data / title
            # (fixes "docubee" queries when Chroma similarity is weak or index stale).
            seen_doc_ids: set = set()
            merged_results: List[Dict[str, Any]] = []
            for r in results:
                merged_results.append(dict(r))
                did = r.get("doc_id")
                if did is not None:
                    try:
                        seen_doc_ids.add(int(did))
                    except (TypeError, ValueError):
                        pass
            try:
                for doc in DatabaseService.find_documents_matching_query_tokens(user_id, user_query):
                    if doc.id not in seen_doc_ids:
                        seen_doc_ids.add(doc.id)
                        merged_results.append({
                            "doc_id": doc.id,
                            "text_entry_id": None,
                            "entry_type": "document",
                            "user_id": user_id,
                            "text": "",
                            "similarity_score": 55.0,
                            "source": "sql_text_match",
                        })
                results = merged_results
                logger.info(
                    "Vector+SQL merge: %s hits total after text fallback",
                    len(results),
                )
            except Exception as e:
                logger.warning("SQL text-match supplement failed (non-fatal): %s", e)

            if not results:
                result = {
                    "success": False,
                    "sql": None,
                    "explanation": "Vector search fallback",
                    "error": error_reason or "No matching documents found",
                    "data": None,
                    "ai_response": "I'm telegram bot i am not able to understand your query"
                }
                if not hasattr(self, '_search_cache'):
                    self._search_cache = {}
                self._search_cache[cache_key] = (time.time(), result)
                return result

            # Format results
            data = []
            for r in results:
                if r.get("entry_type") == "user_text_entry":
                    title = f"Text Entry #{r.get('text_entry_id')} (Score: {r['similarity_score']}%)"
                    item_type = "user_text_entry"
                else:
                    title = f"Doc #{r['doc_id']} (Score: {r['similarity_score']}%)"
                    item_type = "document"
                data.append({
                    "type": item_type,
                    "title": title,
                    "amount": None,
                    "vendor": None,
                    "date": None,
                    "text_entry_id": r.get("text_entry_id"),
                    "_text": r["text"],
                    "_score": r["similarity_score"]
                })

            # Enrich with DB data (same DB as DATABASE_URL — not a separate bot_data.db file)
            try:
                doc_ids = []
                for r in results:
                    if r.get("entry_type") == "user_text_entry":
                        continue
                    did = r.get("doc_id")
                    if did is None:
                        continue
                    try:
                        doc_ids.append(int(did))
                    except (TypeError, ValueError):
                        continue
                doc_details = DatabaseService.fetch_documents_for_vector_enrichment(user_id, doc_ids)

                text_entry_ids = []
                for r in results:
                    if r.get("entry_type") != "user_text_entry":
                        continue
                    tid = r.get("text_entry_id")
                    if tid is None:
                        continue
                    try:
                        text_entry_ids.append(int(tid))
                    except (TypeError, ValueError):
                        continue
                text_entry_details = DatabaseService.fetch_user_text_entries_for_vector_enrichment(
                    user_id, text_entry_ids
                )

                for i, item in enumerate(data):
                    result_row = results[i]
                    if result_row.get("entry_type") == "user_text_entry":
                        teid = result_row.get("text_entry_id")
                        try:
                            teid = int(teid) if teid is not None else None
                        except (TypeError, ValueError):
                            teid = None
                        if teid in text_entry_details:
                            t = text_entry_details[teid]
                            item["entry_kind"] = "user_text_entry"
                            item["type"] = "user_text_entry"
                            item["title"] = f"Expense Note #{teid}"
                            item["amount"] = t.get("amount")
                            item["currency"] = t.get("currency")
                            item["expense_category"] = t.get("expense_category")
                            item["date"] = t.get("created_at")
                            item["user_text_body"] = t.get("text")
                        continue

                    doc_id = result_row.get("doc_id")
                    # Normalize id type (Chroma / drivers may vary)
                    try:
                        doc_key = int(doc_id) if doc_id is not None else None
                    except (TypeError, ValueError):
                        doc_key = None
                    if doc_key is not None and doc_key in doc_details:
                        d = doc_details[doc_key]
                        item["entry_kind"] = "document"
                        item["id"] = d.get("id")
                        item["type"] = d.get("document_type") or "document"
                        item["title"] = d.get("title") or d.get("file_name") or f"Document {doc_key}"
                        item["amount"] = d.get("total_amount")
                        item["vendor"] = d.get("vendor_name")
                        item["date"] = d.get("created_at") or d.get("document_date")
                        item["extracted_data"] = d.get("extracted_data")
                    elif doc_key is not None:
                        logger.warning(
                            "Vector hit doc_id=%s but no DB row for user_id=%s (re-index Chroma or fix DB)",
                            doc_key,
                            user_id,
                        )

            except Exception as e:
                logger.warning(f"Could not enrich vector results: {e}")

            # Filter by score threshold
            MIN_SCORE_THRESHOLD = 50.0
            filtered_data = [d for d in data if d.get('_score', 0) >= MIN_SCORE_THRESHOLD]

            if not filtered_data and data:
                data_sorted = sorted(data, key=lambda x: x.get('_score', 0), reverse=True)
                filtered_data = data_sorted[:1]
                low_confidence = True
            else:
                low_confidence = False
                data_sorted = sorted(filtered_data, key=lambda x: x.get('_score', 0), reverse=True)
                filtered_data = data_sorted[:3]

            # LLM sees DB-backed rows only: denormalized columns + full extracted_data JSON (same user_id as query)
            llm_rows = _rows_for_llm(filtered_data)
            data_summary = json.dumps(llm_rows, indent=2, default=str)
            row_count = len(filtered_data)

            system_prompt = """You are a friendly expense/document assistant.

Answer ONLY using the JSON provided for each hit:
- For documents: read `extracted_data` (full OCR / extraction JSON from the database). Addresses, amounts, vendor names, line items, etc. all live there unless also duplicated in top-level fields.
- For user_text_entry rows: use `text`, `amount`, `expense_category`.

Rules:
1. Answer only what was asked; match the user's language/script.
2. Use amounts/currency as they appear in extracted_data or columns.
3. If extracted_data is missing but extracted_data_raw is present, parse mentally from that string.
4. Do not invent facts not supported by the provided JSON."""

            user_message = f"""Question: "{user_query}"

Retrieved rows for this user (same scope as user_id in DB); total {row_count}:
{data_summary}

Give a direct, conversational answer."""

            llm_result = self.orchestrator.execute_with_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=[],
                temperature=0.5,
                max_tokens=768
            )

            ai_response = llm_result.data.get("content", "") if llm_result.success else "Yeh raha aapka document."
            
            # Add to conversation history
            self._add_to_history(user_id, user_query, ai_response)

            result = {
                "success": True,
                "sql": f"-- VECTOR SEARCH --\n-- Original query: {user_query}\n-- Guardrail: user_id = {user_id}",
                "explanation": f"Semantic search results",
                "error": None,
                "data": filtered_data,
                "row_count": len(filtered_data),
                "columns": ["type", "title", "amount", "vendor", "date"],
                "ai_response": ai_response,
                "fallback": True,
            }

            if not hasattr(self, '_search_cache'):
                self._search_cache = {}
            self._search_cache[cache_key] = (time.time(), result)
            return result

        except Exception as e:
            logger.error(f"Vector search fallback failed: {e}")
            return {
                "success": False,
                "sql": None,
                "error": str(e),
                "data": None,
                "ai_response": "Sorry, I couldn't process your query."
            }


# Singleton instance
_nlp_sql_service_v2 = None


def get_nlp_sql_service_v2() -> NLPSQLServiceV2:
    """Get or create the NLP SQL Service v2 singleton."""
    global _nlp_sql_service_v2
    if _nlp_sql_service_v2 is None:
        _nlp_sql_service_v2 = NLPSQLServiceV2()
    return _nlp_sql_service_v2
