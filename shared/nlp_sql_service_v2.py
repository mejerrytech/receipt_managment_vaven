"""
NLP to SQL Service v2 - Using GPT-4o with Function Calling via Orchestration Layer

This version uses the orchestrator to:
- Call GPT-4o as primary model
- Fallback to Anthropic if needed
- Use function calling for structured outputs
"""

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

_nlp_v2_prompt_cache: Dict[str, tuple[str, str]] = {}


def invalidate_telegram_q_prompt_cache() -> None:
    """Clear cached /q prompts (call after admin updates DB)."""
    _nlp_v2_prompt_cache.clear()


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


def _extract_json_object_from_text(content: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model content that may include markdown or prose."""
    text_content = (content or "").strip()
    if not text_content:
        return None

    if "```json" in text_content:
        text_content = text_content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text_content:
        text_content = text_content.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(text_content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text_content.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(text_content[start:])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _sql_literal(value: str) -> str:
    """Return a single-quoted SQL literal for prompt/guardrail text."""
    return "'" + value.replace("'", "''") + "'"


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


def _documents_to_formatter_rows(documents: List[Any]) -> List[Dict[str, Any]]:
    """Convert Document ORM rows into formatter-ready rows with full OCR JSON."""
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        raw_data = getattr(doc, "extracted_data", None)
        row: Dict[str, Any] = {
            "type": getattr(doc, "document_type", None) or "document",
            "title": getattr(doc, "title", None) or getattr(doc, "file_name", None),
            "amount": getattr(doc, "total_amount", None),
            "vendor": getattr(doc, "vendor_name", None),
            "date": getattr(doc, "document_date", None) or getattr(doc, "created_at", None),
            "raw_data": raw_data,
        }
        parsed = _parse_extracted_json_column(raw_data)
        if parsed is not None:
            row["extracted_data"] = parsed
        rows.append(row)
    return rows


# Function schema for generate_sql function calling
GENERATE_SQL_FUNCTION = {
    "name": "generate_sql_query",
    "description": "Generate a safe SQL query based on the user's natural language request",
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The PostgreSQL SQL query. Must include WHERE user_id filter."
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

# Function schema for parsing multiple expense items from a paragraph
CLASSIFY_MULTI_ITEM_EXPENSE_FUNCTION = {
    "name": "classify_multi_item_expense",
    "description": (
        "Parse a user paragraph that describes multiple purchases/expenses in one message. "
        "Extract every individual item with its amount and category. "
        "Also detect the overall emotional tone. "
        "Return is_multi_item=false if the message is a single expense or not an expense at all."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "is_multi_item": {
                "type": "boolean",
                "description": (
                    "True when the message clearly contains 2+ distinct expense items that "
                    "should each be stored as a separate entry."
                ),
            },
            "items": {
                "type": "array",
                "description": "List of individual expense items extracted from the paragraph.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Short natural-language label for this item (e.g. 'Petrol', 'Milk from KL Reliance')",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Numeric amount in rupees. null if not mentioned.",
                        },
                        "quantity": {
                            "type": "string",
                            "description": "Quantity or unit if mentioned (e.g. '3 kilo', '2 litre'). null otherwise.",
                        },
                        "vendor": {
                            "type": "string",
                            "description": "Vendor / shop name if mentioned. null otherwise.",
                        },
                        "category": {
                            "type": "string",
                            "description": (
                                "Must be exactly one string from the allowed category list "
                                "provided in the system prompt."
                            ),
                        },
                    },
                    "required": ["description", "amount", "category"],
                },
            },
            "user_emotion": {
                "type": "string",
                "enum": ["neutral", "positive", "negative", "stressed_or_urgent", "grateful", "casual"],
                "description": "Overall emotional tone of the message for reply styling.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence 0-1 that this is a multi-item expense paragraph.",
            },
        },
        "required": ["is_multi_item", "items", "user_emotion", "confidence"],
    },
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

# Function schema for resolving follow-up questions using conversation context
RESOLVE_QUERY_WITH_CONTEXT_FUNCTION = {
    "name": "resolve_query_with_context",
    "description": (
        "Rewrite a user question into a self-contained data question using only the "
        "current message and prior conversation context"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "should_use_resolved_query": {
                "type": "boolean",
                "description": "True when the current question depends on prior context or needs spelling/entity normalization",
            },
            "resolved_query": {
                "type": "string",
                "description": "Self-contained question preserving the user's intent and language",
            },
            "focus": {
                "type": "string",
                "description": "Short natural-language focus instruction for the answer formatter",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of what was resolved",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence between 0 and 1",
            },
        },
        "required": [
            "should_use_resolved_query",
            "resolved_query",
            "focus",
            "reason",
            "confidence",
        ],
    },
}

# Function schema for identity / "who am I" routing under user_info intent
CLASSIFY_IDENTITY_PROFILE_FUNCTION = {
    "name": "classify_identity_profile",
    "description": "Decide if the user asks about their own name or Telegram profile identity",
    "parameters": {
        "type": "object",
        "properties": {
            "is_identity_profile_query": {
                "type": "boolean",
                "description": (
                    "True when user asks who they are, their name, or Telegram profile name "
                    "(e.g. who am I, mera naam, kaun hu)"
                ),
            },
            "reason": {
                "type": "string",
                "description": "Short reason for the decision",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence between 0 and 1",
            },
        },
        "required": ["is_identity_profile_query", "reason", "confidence"],
    },
}


class NLPSQLServiceV2:
    """
    NLP to SQL Service using GPT-4o with Function Calling via Orchestration Layer.

    Architecture:
    1. Intent Classification (GPT-4o with function calling)
    2. SQL Generation (GPT-4o with function calling) -> Anthropic fallback
    3. Query Execution (PostgreSQL)
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

    @staticmethod
    def _render_q_user_template(template: Optional[str], **kwargs: Any) -> str:
        out = template or ""
        for k, v in kwargs.items():
            out = out.replace(f"__{k.upper()}__", "" if v is None else str(v))
        return out

    def _q_tpl_pair(self, key: str) -> tuple[str, str]:
        if key in _nlp_v2_prompt_cache:
            return _nlp_v2_prompt_cache[key]
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT system_prompt, user_prompt_template FROM prompts WHERE prompt_key = :k"
                    ),
                    {"k": key},
                ).mappings().first()
        except Exception as e:
            logger.error("Failed to load prompt %s: %s", key, e)
            empty: tuple[str, str] = ("", "")
            _nlp_v2_prompt_cache[key] = empty
            return empty
        if not row:
            logger.error("Missing DB prompt row for key=%s", key)
            empty = ("", "")
            _nlp_v2_prompt_cache[key] = empty
            return empty
        sp = row["system_prompt"] or ""
        raw_upt = row["user_prompt_template"]
        upt = (raw_upt or "") if raw_upt is not None else ""
        tup = (sp, upt)
        _nlp_v2_prompt_cache[key] = tup
        return tup

    def _get_conversation_context(self, user_id: int) -> str:
        """Get formatted conversation history for the user."""
        if user_id not in self.conversation_history:
            return ""

        history = self.conversation_history[user_id]
        if isinstance(history, dict):
            logger.warning("Repairing malformed conversation history for user %s", user_id)
            history = [history] if history.get("query") or history.get("response") else []
            self.conversation_history[user_id] = history
        elif not isinstance(history, list):
            logger.warning(
                "Resetting unsupported conversation history type for user %s: %s",
                user_id,
                type(history).__name__,
            )
            history = []
            self.conversation_history[user_id] = history

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
        history = self.conversation_history.get(user_id)
        if isinstance(history, dict):
            logger.warning("Repairing malformed conversation history before append for user %s", user_id)
            history = [history] if history.get("query") or history.get("response") else []
        elif not isinstance(history, list):
            if history is not None:
                logger.warning(
                    "Resetting unsupported conversation history before append for user %s: %s",
                    user_id,
                    type(history).__name__,
                )
            history = []
            self.conversation_history[user_id] = []

        history.append({
            "query": query,
            "response": response
        })

        self.conversation_history[user_id] = history
        logger.info(f"Added to history for user {user_id}: query='{query[:50]}...', total entries={len(history)}")

        # Keep only last 10
        if len(history) > self.MAX_HISTORY:
            self.conversation_history[user_id] = history[-self.MAX_HISTORY:]

    def clear_user_history(self, user_id: int) -> None:
        """Clear conversation history for a specific user."""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
            logger.info(f"Cleared NLP SQL conversation history for user {user_id}")

    def _resolve_query_with_context(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """Resolve follow-ups through the model instead of local keyword rules."""
        text_query = (user_query or "").strip()
        empty = {
            "effective_query": text_query,
            "focus": "",
            "reason": "",
            "confidence": 0.0,
            "resolved": False,
        }
        if not text_query:
            return empty

        context = self._get_conversation_context(user_id)
        if not context.strip():
            return empty

        system_prompt = """You resolve short, misspelled, or follow-up user questions for a receipt/expense assistant.

Use ONLY the current user message and the previous conversation context. Do not invent vendors, items, prices, categories, or documents.

Your job:
1. If the latest question is already self-contained, keep it unchanged.
2. If it refers to something from context, rewrite it into a self-contained question.
3. Preserve the user's language/script and intent.
4. Normalize spelling only when the context clearly supports it. Common Hinglish typos like "merta" should usually mean "mera" (my), not a place/vendor/entity, unless context explicitly established Merta as a saved entity.
5. For item questions, keep the requested item, quantity/unit-price/total-price distinction, and target document/vendor if known.
6. For vendor/entity questions, do not assume aliases unless the context already established that relationship.
7. For location follow-ups like "sirf Lucknow se", "or Delhi se", preserve the previous item/category (for example petrol/fuel) and add the requested city/location as a hard filter.
8. If the latest question is "kaise", "kese", "how", "breakdown", or asks why/how a previous total was calculated, rewrite it as a breakdown/explanation of the immediately previous total using the same category/entity from context.
9. If the previous context is a saved manual expense ("Saved as Travel/Shopping/etc expense"), follow-up questions like "kaha gya tha me", "flight se kaha gya", "kya kya liya", or "kb liya" should resolve to the saved manual expense text and category, not the user's Telegram profile.
10. Prefer the immediately previous expense/category answer over older receipt/document matches when resolving short follow-ups.
11. Return a short focus instruction that the final answer can use to avoid unrelated rows.

Call resolve_query_with_context only."""
        user_message = f"""Latest user question:
\"\"\"{text_query}\"\"\"

Conversation context:
{context}

Resolve now."""

        try:
            result = self.orchestrator.execute_with_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=[RESOLVE_QUERY_WITH_CONTEXT_FUNCTION],
                temperature=0.1,
                max_tokens=500,
            )
            if result.success and result.function_calls:
                func_call = result.function_calls[0]
                args = func_call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                confidence = float(args.get("confidence", 0.0) or 0.0)
                resolved_query = (args.get("resolved_query") or text_query).strip()
                should_resolve = bool(args.get("should_use_resolved_query", False))
                if should_resolve and resolved_query and confidence >= 0.55:
                    return {
                        "effective_query": resolved_query,
                        "focus": (args.get("focus") or "").strip(),
                        "reason": (args.get("reason") or "").strip(),
                        "confidence": confidence,
                        "resolved": True,
                    }
                return {
                    **empty,
                    "focus": (args.get("focus") or "").strip(),
                    "reason": (args.get("reason") or "").strip(),
                    "confidence": confidence,
                }
        except Exception as e:
            logger.warning("Context query resolver failed; using original query: %s", e)

        return empty

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

    def _should_use_global_summary_sql(self, user_query: str, user_id: int) -> bool:
        """
        Dynamically decide summary routing via prompt+function-calling.
        If the model is unavailable, keep the normal hybrid path.
        """
        context = self._get_conversation_context(user_id)
        sys_t, usr_t = self._q_tpl_pair("q_telegram_summary_routing")
        system_prompt = sys_t + """

Runtime rule:
- Return true only for broad overall expense totals/overview.
- Return false for category breakdowns/counts/rankings, item questions, vendor-specific questions, or follow-ups about one bill/document."""
        user_message = self._render_q_user_template(
            usr_t,
            user_query=user_query,
            context=context,
        )

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
            logger.warning("Summary routing classifier failed; using hybrid path: %s", e)

        return False

    def _is_identity_profile_query(self, user_query: str, user_id: int) -> bool:
        """Decide identity/name asks via DB prompt."""
        q = (user_query or "").strip()
        if not q:
            return False

        context = self._get_conversation_context(user_id)
        sys_t, usr_t = self._q_tpl_pair("q_telegram_identity_profile")
        system_prompt = sys_t
        user_message = self._render_q_user_template(
            usr_t,
            user_query=user_query,
            context=context,
        )

        try:
            result = self.orchestrator.execute_with_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=[CLASSIFY_IDENTITY_PROFILE_FUNCTION],
                temperature=0.1,
                max_tokens=300,
            )
            if result.success and result.function_calls:
                func_call = result.function_calls[0]
                args = func_call["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                decision = bool(args.get("is_identity_profile_query", False))
                confidence = float(args.get("confidence", 0.0) or 0.0)
                if confidence >= 0.60:
                    return decision
        except Exception as e:
            logger.warning("Identity profile classifier failed; using default route: %s", e)

        return False

    def _understand_intent(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """
        Step 1: Understand user intent using GPT-4o with function calling.
        """
        context = self._get_conversation_context(user_id)
        sys_t, usr_t = self._q_tpl_pair("q_telegram_intent")
        system_prompt = sys_t
        user_message = self._render_q_user_template(
            usr_t,
            user_query=user_query,
            context=context,
        )

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
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

    def generate_expense_save_confirmation(
        self,
        saved_items: List[Dict[str, Any]],
        user_emotion: str = "neutral",
        user_id: Optional[int] = None,
    ) -> str:
        """
        Generate a friendly, emoji-rich confirmation message after saving expense entries.
        Works for both single-item and multi-item saves.
        Emoji placement rules are fully driven by the DB prompt — no static formatting here.
        Falls back to a simple plain string only if the AI call fails.
        """
        if not saved_items:
            return "Koi expense save nahi hua."

        emotion = user_emotion or "neutral"
        total = sum(
            float(it.get("amount") or 0) for it in saved_items if it.get("amount") is not None
        )
        total_str = f"₹{total:,.0f}" if total > 0 else "N/A"

        sys_t, usr_t = self._q_tpl_pair("q_expense_save_confirm")
        if not sys_t:
            # Prompt not yet in DB — return a simple fallback
            if len(saved_items) == 1:
                it = saved_items[0]
                amt = it.get("amount")
                cat = it.get("category") or "Other"
                amt_str = f" ₹{float(amt):,.0f}" if amt is not None else ""
                return f"{cat} expense{amt_str} saved ✅"
            return f"{len(saved_items)} expenses saved ✅ — Total: {total_str}"

        items_json = json.dumps(saved_items, ensure_ascii=False)
        user_message = self._render_q_user_template(
            usr_t,
            user_emotion=emotion,
            saved_items=items_json,
            total_amount=total_str,
        )

        result = self.orchestrator.execute_with_fallback(
            system_prompt=sys_t,
            user_message=user_message,
            functions=[],
            temperature=0.6,
            max_tokens=400,
        )

        if result.success and result.data and result.data.get("content"):
            return result.data["content"].strip()

        # Plain fallback if AI unavailable
        if len(saved_items) == 1:
            it = saved_items[0]
            amt = it.get("amount")
            cat = it.get("category") or "Other"
            amt_str = f" ₹{float(amt):,.0f}" if amt is not None else ""
            return f"{cat} expense{amt_str} saved ✅"
        return f"{len(saved_items)} expenses saved ✅ — Total: {total_str}"

    def classify_multi_item_expense(self, user_text: str) -> Dict[str, Any]:
        """
        Detect whether a plain user message describes multiple distinct expense items
        (e.g. a shopping paragraph). If yes, return each item with its amount and category
        so the caller can persist and display them individually.

        Returns a dict:
        {
            "is_multi_item": bool,
            "items": [{"description", "amount", "quantity", "vendor", "category"}, ...],
            "user_emotion": str,
            "confidence": float,
        }
        On any failure returns is_multi_item=False so caller falls back gracefully.
        """
        text = (user_text or "").strip()
        default = {
            "is_multi_item": False,
            "items": [],
            "user_emotion": "neutral",
            "confidence": 0.0,
        }
        if not text:
            return default

        category_lines = "\n".join(f"- {c}" for c in self.expense_categories)
        system_prompt = f"""You are an expense-extraction assistant for a receipt/expense bot.

A user can send a single paragraph describing MULTIPLE purchases made in one go, for example:
"Maine 2000 ka petrol dalwaya. Maine KL Reliance se 66 ka milk, 140 ka tel, 238 ke aam liye."

ALLOWED expense categories (use the EXACT string from this list):
{category_lines}

Your tasks:
1. Decide if the message contains 2 or more DISTINCT expense items (is_multi_item).
   - A single purchase like "Maine 500 ka petrol dalwaya" → is_multi_item = false.
   - Two or more purchases in one message → is_multi_item = true.
2. For each item extract:
   - description  : short label in the user's language (Hindi/English/Hinglish OK)
   - amount       : numeric rupee value (null if not stated)
   - quantity     : quantity/unit if mentioned, otherwise null
   - vendor       : shop/brand/vendor name if mentioned, otherwise null
   - category     : pick the single best category from the ALLOWED list above
3. user_emotion   : overall tone of the whole message
4. confidence     : how confident you are this is a multi-item expense paragraph (0–1)

Critical rules:
- NEVER invent items or amounts not present in the text.
- If is_multi_item is false, items can be empty [].
- category MUST be one of the allowed strings exactly (use 'Other' if uncertain).
- Understand intent — do NOT rely on commas or conjunctions alone; understand what the user is saying.
- Call classify_multi_item_expense only."""

        user_message = f"""User message:
\"\"\"{text}\"\"\"

Parse now."""

        try:
            result = self.orchestrator.execute_with_fallback(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=[CLASSIFY_MULTI_ITEM_EXPENSE_FUNCTION],
                temperature=0.1,
                max_tokens=1024,
            )
            if result.success and result.function_calls:
                func_call = result.function_calls[0]
                args = func_call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)

                is_multi = bool(args.get("is_multi_item", False))
                confidence = float(args.get("confidence", 0.0) or 0.0)
                emotion = args.get("user_emotion") or "neutral"
                if emotion not in (
                    "neutral", "positive", "negative",
                    "stressed_or_urgent", "grateful", "casual",
                ):
                    emotion = "neutral"

                # Only trust multi-item with reasonable confidence
                if not is_multi or confidence < 0.55:
                    return {**default, "user_emotion": emotion, "confidence": confidence}

                raw_items = args.get("items") or []
                items: List[Dict[str, Any]] = []
                for it in raw_items:
                    if not isinstance(it, dict):
                        continue
                    desc = (it.get("description") or "").strip()
                    if not desc:
                        continue
                    amt_raw = it.get("amount")
                    try:
                        amt = float(amt_raw) if amt_raw is not None else None
                    except (TypeError, ValueError):
                        amt = None
                    cat = self._normalize_expense_category_label(it.get("category"))
                    items.append({
                        "description": desc,
                        "amount": amt,
                        "quantity": (it.get("quantity") or "").strip() or None,
                        "vendor": (it.get("vendor") or "").strip() or None,
                        "category": cat,
                    })

                if len(items) < 2:
                    # Fewer than 2 valid items → not really multi-item
                    return {**default, "user_emotion": emotion, "confidence": confidence}

                logger.info(
                    "Multi-item expense detected: %d items, confidence=%.2f", len(items), confidence
                )
                return {
                    "is_multi_item": True,
                    "items": items,
                    "user_emotion": emotion,
                    "confidence": confidence,
                }
        except Exception:
            logger.exception("Multi-item expense classifier failed; falling back to single-item path")

        return default

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

    def _guardrail_username(self, user_id: int) -> str:
        """Return the stored username for the current DB user, or a stable internal fallback."""
        try:
            user = DatabaseService.get_user_by_id(user_id)
            username = (getattr(user, "username", None) or "").strip() if user else ""
            return username or f"user_id:{user_id}"
        except Exception:
            logger.exception("Could not load username guardrail for user_id=%s", user_id)
            return f"user_id:{user_id}"

    def _sql_has_current_user_guardrail(self, sql: str, user_id: int, username: Optional[str] = None) -> bool:
        """Accept only SELECT SQL scoped to the current owner identity."""
        if not sql:
            return False

        sql_norm = " ".join(sql.split())
        sql_compact = sql_norm.replace(" ", "").lower()
        if not sql_norm.upper().startswith("SELECT"):
            return False

        uid = str(user_id)
        has_user_id_scope = (
            f"user_id={uid}" in sql_compact
            or f"users.id={uid}" in sql_compact
            or f".user_id={uid}" in sql_compact
            or (
                " from users " in f" {sql_norm.lower()} "
                and f"id={uid}" in sql_compact
            )
        )

        uname = (username or self._guardrail_username(user_id) or "").strip()
        has_username_scope = False
        if uname and not uname.startswith("user_id:"):
            uname_escaped = re.escape(uname.lower())
            has_username_scope = bool(
                re.search(
                    rf"(?:\b\w+\.)?username\s*=\s*['\"]{uname_escaped}['\"]",
                    sql_norm.lower(),
                )
            )

        return has_user_id_scope or has_username_scope

    def generate_sql(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """
        Step 2: Generate SQL using GPT-4o with function calling.
        Falls back to Anthropic if GPT-4o fails.
        """
        username = self._guardrail_username(user_id)
        schema = self._q_tpl_pair("q_telegram_db_schema")[0]
        sys_tpl, usr_t = self._q_tpl_pair("q_telegram_generate_sql")
        system_prompt = (
            sys_tpl.replace("__DB_SCHEMA__", schema).replace("__USER_ID__", str(user_id))
        )
        username_guardrail = (
            f"users.username = {_sql_literal(username)}"
            if username and not username.startswith("user_id:")
            else f"users.id = {user_id}"
        )
        system_prompt += f"""

Runtime grounding rules:
- Prefer the latest resolved question and conversation context over isolated word matches.
- For item, product, quantity, unit-price, or line-item questions, include documents.extracted_data as raw_data plus vendor/title/amount/date so the formatter can inspect nested OCR JSON.
- For category breakdowns, category counts, or "highest expense by category" style questions, aggregate documents.expense_category and user_text_entries.expense_category together when relevant.
- If the user asks for Telegram vs WhatsApp data, filter documents.source and user_text_entries.source using 'telegram' or 'whatsapp'.
- For "kaise/how/breakdown" follow-ups after a total, return the contributing rows (title/text, amount, category, date) instead of another total-only aggregate.
- For manual expense follow-ups ("kya kya liya", "kb liya", "kaha gya", "flight se kaha") return user_text_entries.text, amount, expense_category, created_at so the formatter can infer details from the saved text.
- For category totals like shopping/travel, include user_text_entries and documents only when the saved category matches. Do not use unrelated receipt items just because they are in conversation history.
- Location/city words such as Lucknow, Delhi/Dehli, Jaipur, Mumbai, etc. are hard filters. For these, search documents.vendor_name, documents.title, documents.raw_text, and documents.extracted_data for the location while also preserving the requested item/category from context.
- If no row supports the requested location/entity, return zero rows rather than reusing a previous or semantically similar row.
- For petrol/fuel questions, do not require the literal word "petrol" in OCR items. Fuel-station evidence includes documents.expense_category = 'Fuel', vendor/title/raw_text/extracted_data containing fuel, fuels, petrol, diesel, oil, IndianOil, IOCL, HPCL, BPCL, pump, or filling station.
- For petrol/fuel amount questions, return the receipt total_amount/amount for matching fuel receipts. If a location is requested, apply BOTH the fuel evidence and the location evidence in the WHERE clause.
- For named vendor/entity questions, constrain results to rows whose saved fields or extracted OCR JSON support that entity. Do not assume aliases or marketplace relationships unless the query/context explicitly states them.
- If the vendor/entity phrase appears misspelled or contains a generic business type, search saved fields and extracted OCR JSON using the distinctive part(s) of the phrase rather than requiring the entire phrase to match exactly.
- If exact structured SQL is uncertain, return rows with raw_data instead of collapsing to total_amount only.
- Current authenticated owner guardrail is {username_guardrail}; internal owner key is user_id = {user_id}.
- Keep every query SELECT-only and scoped to this exact current owner. Prefer joining users and filtering {username_guardrail}; user_id = {user_id} is also accepted as the internal owner key.
- Never answer with rows for any other username/user_id, even if the user asks for someone else's data."""
        context = self._get_conversation_context(user_id)
        user_message = self._render_q_user_template(
            usr_t,
            user_query=user_query,
            context=context,
        )
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
                    if not self._sql_has_current_user_guardrail(sql, user_id, username):
                        return {
                            "sql": None,
                            "explanation": None,
                            "is_safe": False,
                            "error": f"Security error: Query must be SELECT-only and scoped to username={username}",
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
            parsed = _extract_json_object_from_text(content)
            if parsed:
                sql = parsed.get("sql", "")
                if sql:
                    if not self._sql_has_current_user_guardrail(sql, user_id, username):
                        return {
                            "sql": None,
                            "explanation": None,
                            "is_safe": False,
                            "error": f"Security error: Query must be SELECT-only and scoped to username={username}",
                            "provider_used": result.provider_used.value
                        }

                return {
                    "sql": sql,
                    "explanation": parsed.get("explanation", ""),
                    "is_safe": parsed.get("is_safe", False),
                    "error": parsed.get("error"),
                    "provider_used": result.provider_used.value
                }
            if content and content.strip():
                logger.warning("SQL fallback content was not JSON; ignoring provider content.")

        # Both models failed
        return {
            "sql": None,
            "explanation": None,
            "is_safe": False,
            "error": result.error or "Failed to generate SQL",
            "provider_used": "failed"
        }

    def execute_query(self, sql: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Execute SQL against PostgreSQL and re-validate current-user guardrails."""
        if user_id is not None and sql:
            username = self._guardrail_username(user_id)
            if not self._sql_has_current_user_guardrail(sql, user_id, username):
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "error": f"Security error: query must be SELECT and scoped to username={username}",
                }
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

        sys_t, usr_t = self._q_tpl_pair("q_telegram_social_reply")
        system_prompt = sys_t
        user_message = self._render_q_user_template(
            usr_t,
            intent_hint=intent_hint,
            user_query=user_query,
        )

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

    @staticmethod
    def _user_display_name(user: Any) -> Optional[str]:
        parts = []
        if getattr(user, "first_name", None):
            parts.append(str(user.first_name).strip())
        if getattr(user, "last_name", None):
            parts.append(str(user.last_name).strip())
        name = " ".join(p for p in parts if p)
        if name:
            return name
        username = getattr(user, "username", None)
        if username:
            return f"@{username}"
        return None

    def _build_identity_profile_response(self, user_query: str, user_id: int) -> str:
        user = DatabaseService.get_user_by_id(user_id)
        if not user:
            return "User profile nahi mila."

        display = self._user_display_name(user)
        hindi = bool(re.search(r"[\u0900-\u097F]", user_query)) or bool(
            re.search(r"(?i)\b(kon|kaun|mera|naam|hu|hoon|aap)\b", user_query)
        )

        if display:
            if hindi:
                return f"Aap {display} hain — ye aapka Telegram profile naam hai."
            return f"You are {display} — that's your Telegram profile name."

        if hindi:
            return (
                "Aapka naam abhi database me save nahi hai. "
                "Telegram me profile naam set karein, phir dubara try karein."
            )
        return (
            "Your name isn't saved yet. Set your name on Telegram and try again."
        )

    def ask_ai(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """
        Complete pipeline with orchestration:
        1. Understand Intent (GPT-4o)
        2. Route to appropriate handler
        3. Generate SQL (GPT-4o) or fallback to Vector Search
        4. Execute Query
        5. Format Response (GPT-4o)
        """
        # Step 1: Intent Classification
        intent_analysis = self._understand_intent(user_query, user_id)
        logger.info(f"Intent analysis for '{user_query}': {intent_analysis}")

        intent = intent_analysis.get("intent", "unknown")
        confidence = intent_analysis.get("confidence", 0)

        # Document/data questions use PostgreSQL + Chroma (hybrid), scoped by user_id.
        # Route user_info here too; short Hinglish follow-ups like "kaha gya tha me"
        # are often expense-context questions, not profile questions.
        if intent in ("sql_query", "semantic_search", "unknown", "user_info") or confidence < 0.6:
            logger.info(
                "Hybrid SQL+vector routing (intent=%s, confidence=%s)",
                intent,
                confidence,
            )
            return self._hybrid_sql_and_vector_query(
                user_query,
                user_id,
                route_reason=f"intent={intent},confidence={confidence}",
            )

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

        # Any other intent: hybrid retrieval as safe default.
        return self._hybrid_sql_and_vector_query(user_query, user_id, route_reason=f"intent={intent}")

    def _hybrid_sql_and_vector_query(
        self, user_query: str, user_id: int, route_reason: str = ""
    ) -> Dict[str, Any]:
        """Answer using PostgreSQL (SQL) and Chroma (vector), merged for the same user_id."""
        resolution = self._resolve_query_with_context(user_query, user_id)
        effective_query = resolution["effective_query"]
        focus = resolution.get("focus", "")
        vector_items = self._retrieve_vector_hits(effective_query, user_id)
        sql_result = self.generate_sql(effective_query, user_id)
        sql = sql_result.get("sql") if sql_result.get("is_safe") else None
        sql_rows: List[Dict[str, Any]] = []
        exec_error: Optional[str] = None
        exec_result: Dict[str, Any] = {"columns": [], "rows": [], "row_count": 0}

        if sql:
            exec_result = self.execute_query(sql, user_id=user_id)
            exec_error = exec_result.get("error")
            if not exec_error:
                sql_rows = exec_result.get("rows") or []

        if sql_rows:
            ai_response = self._format_response(
                user_query,
                sql or "",
                sql_rows,
                user_id,
                vector_supplement=vector_items,
                resolved_query=effective_query,
                focus=focus,
            )
            self._add_to_history(user_id, user_query, ai_response)
            try:
                self._store_sql_results(effective_query, sql or "", sql_rows, user_id)
            except Exception as e:
                logger.warning("Failed to store SQL results in vector DB (non-critical): %s", e)
            return {
                "success": True,
                "sql": sql,
                "explanation": sql_result.get("explanation"),
                "error": None,
                "data": sql_rows,
                "row_count": len(sql_rows),
                "columns": exec_result.get("columns") if sql else [],
                "ai_response": ai_response,
                "provider_used": sql_result.get("provider_used", "unknown"),
                "hybrid": True,
                "vector_hits": len(vector_items),
            }

        if vector_items:
            return self._answer_from_vector_hits(
                user_query,
                user_id,
                vector_items,
                sql_error=exec_error or sql_result.get("error") or route_reason,
                resolved_query=effective_query,
                focus=focus,
            )

        return {
            "success": False,
            "sql": sql,
            "explanation": sql_result.get("explanation"),
            "error": exec_error or sql_result.get("error") or route_reason,
            "data": None,
            "ai_response": "No matching data found in your documents for this question.",
        }

    def _format_response(
        self,
        user_query: str,
        sql: str,
        data: List[Dict],
        user_id: int,
        vector_supplement: Optional[List[Dict[str, Any]]] = None,
        resolved_query: Optional[str] = None,
        focus: str = "",
    ) -> str:
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

        formatter_limit = 100
        formatter_rows = data[:formatter_limit]
        data_summary = (
            json.dumps(formatter_rows, indent=2, default=str) if data else "[]"
        )
        if vector_supplement:
            vec_rows = _rows_for_llm(vector_supplement[:5])
            data_summary += (
                "\n\nAdditional semantic matches from vector DB (same user only):\n"
                + json.dumps(vec_rows, indent=2, default=str)
            )
        row_count = len(data)

        sys_t, usr_t = self._q_tpl_pair("q_telegram_format_sql_response")
        system_prompt = sys_t + """

Runtime grounding rules:
- Use only the provided rows and previous conversation context. If a requested vendor/entity/item is not supported by the rows, say it was not found.
- Requested city/location/entity words are hard filters. Do not answer with a row unless that row's vendor/title/raw_data/extracted_data/text explicitly supports the requested location/entity.
- If the user asks "or Delhi/Dehli se" after a Lucknow answer, keep the same item/category context but require Delhi/Dehli support in the row; otherwise say it was not found.
- For petrol/fuel questions, fuel-station receipts count as petrol/fuel evidence even if OCR item names are generic like "Product 1". Use vendor/title/category/raw_data terms such as Fuel/Fuels, Petrol, Diesel, Oil, IndianOil, IOCL, HPCL, BPCL, pump, or filling station.
- If the row supports the requested location and is a fuel-station receipt, use its total amount for "kitne ka fill karwaya" style questions.
- Treat raw_data, extracted_data, extracted_data_raw, and nested OCR JSON as first-class answer data.
- Treat user_text_entries.text as first-class answer data for manual expenses. If the user asks where they went, what they bought, or when, infer it from that saved text and its created_at only.
- For item questions, inspect item description/name plus quantity, unit price/price, amount/total. Do not answer with receipt total unless the user asked for the whole bill total.
- For follow-up questions, honor the resolved question/focus below over broad semantic matches.
- Do not transfer items from one vendor/document to another unless the data explicitly supports that relationship."""
        context = self._get_conversation_context(user_id)
        resolved_line = ""
        if resolved_query and resolved_query.strip() and resolved_query.strip() != user_query.strip():
            resolved_line = f"\nResolved question from context: \"{resolved_query.strip()}\""
        focus_line = f"\nAnswer focus: {focus.strip()}" if focus.strip() else ""
        user_message = self._render_q_user_template(
            usr_t,
            user_query=user_query + resolved_line + focus_line,
            row_count=str(row_count),
            data_summary=data_summary,
            context=context,
        )

        result = self.orchestrator.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
            functions=[],  # No function calling needed for formatting
            temperature=0.5,
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

            # Use the shared OpenAI embedding path so dimensions/model stay consistent.
            embedding = self.vector_service._generate_embedding(searchable_text)

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

    def _retrieve_vector_hits(
        self, user_query: str, user_id: int, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Chroma semantic search + SQL token match, enriched from PostgreSQL (user_id scoped)."""
        username = self._guardrail_username(user_id)
        results = self.vector_service.search(
            query=user_query,
            user_id=user_id,
            n_results=n_results,
            username=username if not username.startswith("user_id:") else None,
        )
        results = [
            r for r in results
            if int(r.get("user_id", user_id)) == int(user_id)
            and (
                not r.get("username")
                or username.startswith("user_id:")
                or r.get("username") == username
            )
        ]
        logger.info(
            "Vector guardrail retained %s hits for username=%s user_id=%s",
            len(results),
            username,
            user_id,
        )

        if not results:
            return []

        data: List[Dict[str, Any]] = []
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
                "_text": r.get("text", ""),
                "_score": r.get("similarity_score", 0),
            })

        try:
            doc_ids: List[int] = []
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

            text_entry_ids: List[int] = []
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
        except Exception as e:
            logger.warning("Could not enrich vector results: %s", e)

        min_score = 50.0
        filtered = [d for d in data if d.get("_score", 0) >= min_score]
        if not filtered and data:
            filtered = sorted(data, key=lambda x: x.get("_score", 0), reverse=True)[:1]
        else:
            filtered = sorted(filtered, key=lambda x: x.get("_score", 0), reverse=True)[:3]
        return filtered

    def _answer_from_vector_hits(
        self,
        user_query: str,
        user_id: int,
        vector_items: List[Dict[str, Any]],
        sql_error: Optional[str] = None,
        resolved_query: Optional[str] = None,
        focus: str = "",
    ) -> Dict[str, Any]:
        """Format an answer from vector hits (already user-scoped)."""
        llm_rows = _rows_for_llm(vector_items)
        data_summary = json.dumps(llm_rows, indent=2, default=str)
        sys_t, usr_t = self._q_tpl_pair("q_telegram_vector_semantic")
        sys_t += """

Runtime grounding rules:
- Use only the retrieved rows and conversation context. If the requested vendor/entity/item is not supported by these rows, say it was not found.
- Requested city/location/entity words are hard filters. Do not answer with a retrieved row unless its text/vendor/title/extracted data explicitly supports that location/entity.
- If a location follow-up asks for another city (for example Delhi/Dehli after Lucknow), keep the item/category context but require that new city in the retrieved row.
- For petrol/fuel questions, fuel-station receipts count as petrol/fuel evidence even when OCR item names are generic like "Product 1". Use the receipt total for "kitne ka fill karwaya" when the retrieved row supports the requested location.
- Treat extracted_data, extracted_data_raw, raw_data, and nested OCR JSON as first-class answer data.
- Treat user_text_body/text from manual expenses as first-class answer data. Use it for saved purchase/travel details and follow-ups.
- For item questions, inspect item description/name plus quantity, unit price/price, amount/total. Do not answer with receipt total unless the user asked for the whole bill total.
- Honor the resolved question/focus below and ignore unrelated semantic matches."""
        hist = self._get_conversation_context(user_id)
        resolved_line = ""
        if resolved_query and resolved_query.strip() and resolved_query.strip() != user_query.strip():
            resolved_line = f"\nResolved question from context: \"{resolved_query.strip()}\""
        focus_line = f"\nAnswer focus: {focus.strip()}" if focus.strip() else ""
        user_message = self._render_q_user_template(
            usr_t,
            user_query=user_query + resolved_line + focus_line,
            data_summary=data_summary,
            row_count=str(len(vector_items)),
            context=hist,
        )
        llm_result = self.orchestrator.execute_with_fallback(
            system_prompt=sys_t,
            user_message=user_message,
            functions=[],
            temperature=0.5,
            max_tokens=768,
        )
        ai_response = (
            llm_result.data.get("content", "").strip()
            if llm_result.success and llm_result.data
            else "Yeh raha aapka document."
        )
        self._add_to_history(user_id, user_query, ai_response)
        return {
            "success": True,
            "sql": f"-- HYBRID VECTOR --\n-- user_id = {user_id}",
            "explanation": "Semantic search (vector DB + SQL enrichment)",
            "error": sql_error,
            "data": vector_items,
            "row_count": len(vector_items),
            "columns": ["type", "title", "amount", "vendor", "date"],
            "ai_response": ai_response,
            "fallback": True,
            "hybrid": True,
        }

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
            vector_items = self._retrieve_vector_hits(user_query, user_id)
            if not vector_items:
                result = {
                    "success": False,
                    "sql": None,
                    "explanation": "Vector search fallback",
                    "error": error_reason or "No matching documents found",
                    "data": None,
                    "ai_response": "I'm telegram bot i am not able to understand your query",
                }
                if not hasattr(self, '_search_cache'):
                    self._search_cache = {}
                self._search_cache[cache_key] = (time.time(), result)
                return result

            result = self._answer_from_vector_hits(
                user_query, user_id, vector_items, sql_error=error_reason
            )
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
