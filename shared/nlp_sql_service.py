"""NLP to SQL Service - Convert natural language to SQL queries safely"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import openai
import anthropic
from sqlalchemy import text

load_dotenv()

logger = logging.getLogger("nlp_sql")

# Import vector service for fallback and embeddings
from shared.vector_service import get_vector_service
from shared.database import engine, DatabaseService

# Model configuration
GPT4O_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-opus-4-5-20251101"

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

Key notes:
- Always filter by user_id for security
- extracted_data contains full JSON from OCR
- document_type examples: invoice, receipt, product_listing, etc.
- Amounts are in the currency specified (mostly INR)
"""


class NLPSQLService:
    """Service to convert natural language to SQL queries."""

    def __init__(self):
        # Initialize OpenAI client (GPT-4o primary)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.error("OPENAI_API_KEY not set!")
        self.openai_client = openai.OpenAI(api_key=openai_key)
        # Initialize Claude client (fallback)
        self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Vector service
        self.vector_service = get_vector_service()

    def _understand_intent(self, user_query: str) -> Dict[str, Any]:
        """
        First step: Understand what the user actually wants.

        Analyzes the query to determine:
        - intent_type: 'sql_query', 'semantic_search', 'conversation', 'greeting', 'unknown'
        - needs: What information user is looking for
        - confidence: How clear the intent is (0-1)

        Returns:
            Dict with intent analysis
        """
        system_prompt = """You are an intent classifier for a document management bot.

Analyze the user's query and classify the intent. Respond in JSON format.

Intent Types:
- "sql_query": User wants specific data from their documents (show invoices, find receipts, totals, etc.)
- "semantic_search": User asks about content/topics in documents (tell me about AI, what do I have about X, etc.)
- "conversation": General chat, questions about the bot, help, etc.
- "greeting": Hello, hi, thanks, bye, etc.
- "user_info": User asks about their profile, name, settings
- "unknown": Unclear what user wants

Examples:
User: "Show my invoices" → {"intent": "sql_query", "needs": "list of invoice documents", "confidence": 0.95}
User: "Tell me about AI" → {"intent": "semantic_search", "needs": "find documents related to AI topic", "confidence": 0.9}
User: "Give me summary of my documents" → {"intent": "sql_query", "needs": "aggregate statistics of all documents", "confidence": 0.9}
User: "Give me summary of akash enterprises" → {"intent": "semantic_search", "needs": "find documents about akash enterprises entity", "confidence": 0.9}
User: "Total amount across all my receipts" → {"intent": "sql_query", "needs": "sum of amounts for receipt documents", "confidence": 0.9}
User: "What is my name?" → {"intent": "user_info", "needs": "user profile data", "confidence": 0.95}
User: "Hello" → {"intent": "greeting", "needs": "greeting response", "confidence": 0.99}
User: "How do I use this bot?" → {"intent": "conversation", "needs": "help/instructions", "confidence": 0.85}
User: "random gibberish" → {"intent": "unknown", "needs": "clarification", "confidence": 0.3}
User: "Show me all my invoices" → {"intent": "sql_query", "needs": "list of invoice documents", "confidence": 0.95}
User: "Show me all my receipts" → {"intent": "sql_query", "needs": "list of receipt documents", "confidence": 0.95}
User : "Give me all dollar invoices" → {"intent": "sql_query", "needs": "list of invoice documents with currency USD", "confidence": 0.95}

IMPORTANT RULES:
- "summary of [entity/company/topic]" = semantic_search (find docs about that topic)
- "summary of my documents" = sql_query (aggregate stats across all docs)
- "summary" alone = sql_query (user's overall document statistics)

Respond ONLY with valid JSON in this exact format:
{
    "intent": "sql_query|semantic_search|conversation|greeting|user_info|unknown",
    "needs": "description of what user wants",
    "confidence": 0.0-1.0,
    "suggested_action": "generate_sql|vector_fallback|chat_response|greet|explain_bot"
}"""

        # Try GPT-4o first
        try:
            prompt = f"{system_prompt}\n\nClassify this query: {user_query}"
            response = self.openai_client.chat.completions.create(
                model=GPT4O_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            
            text = response.choices[0].message.content
            # Clean up JSON if needed
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            return json.loads(text)
        except Exception as e:
            logger.warning(f"GPT-4o intent classification failed: {e}, trying Claude")
        
        # Fallback to Claude
        try:
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Classify this query: {user_query}"}
                ]
            )
            
            text = response.content[0].text if hasattr(response, 'content') else str(response)
            # Clean up JSON if needed
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            return json.loads(text)
        except Exception as e:
            logger.error(f"Claude intent classification also failed: {e}")
        
        return {
            "intent": "unknown",
            "needs": "could not understand query",
            "confidence": 0.0,
            "suggested_action": "vector_fallback"
        }

    def generate_sql(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """
        Convert natural language query to SQL with user-specific guardrails.

        Args:
            user_query: The natural language question
            user_id: The user's database ID (for security filtering)

        Returns:
            Dict with 'sql', 'explanation', 'is_safe', 'error'
        """
        system_prompt = f"""
You are an AI assistant that converts natural language questions into safe PostgreSQL SQL queries.

DATABASE SCHEMA:
{DB_SCHEMA}

========================
CRITICAL SECURITY RULES:
========================
1. ALWAYS include "WHERE user_id = {user_id}" in the query (MANDATORY)
2. NEVER access data of other users
3. ONLY generate SELECT queries (NO INSERT, UPDATE, DELETE, DROP, ALTER)
4. "my" always refers to user_id = {user_id}
5. Return only valid PostgreSQL SQL

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
   - Example:

   SELECT DISTINCT
       vendor_name AS title,
       'vendor' AS type,
       NULL AS amount,
       vendor_name AS vendor,
       NULL AS date
   FROM documents
   WHERE user_id = {user_id} AND vendor_name IS NOT NULL


2. If query is about totals (SUM, COUNT, etc.)
   - Return aggregated values with proper aliases
   - Example:
     SELECT SUM(total_amount) AS amount, currency
     FROM documents
     WHERE user_id = {user_id}

3. If user asks about their profile/details (e.g., "what is my name", "my username", "who am I"):
   - Query the users table
   - Example:
     SELECT first_name, last_name, username, telegram_id FROM users WHERE id = {user_id}

4. If data is missing:
   - title → fallback to file_name
   - type → fallback to 'unknown'
   - amount → can be NULL

========================
EXAMPLES:
========================

User: "what is my name"
Output:
{{
    "sql": "SELECT first_name, last_name, username FROM users WHERE id = {user_id}",
    "explanation": "Retrieves the user's name and profile info",
    "is_safe": true,
    "error": null
}}

User: "Show me my documents"
Output:
{{
    "sql": "SELECT document_type AS type, title, total_amount AS amount, vendor_name AS vendor, created_at AS date FROM documents WHERE user_id = {user_id} ORDER BY created_at DESC",
    "explanation": "Retrieves all documents for the user",
    "is_safe": true,
    "error": null
}}

User: "Show invoices from last month"
Output:
{{
    "sql": "SELECT document_type AS type, title, total_amount AS amount, vendor_name AS vendor, created_at AS date FROM documents WHERE user_id = {user_id} AND document_type ILIKE '%invoice%' AND created_at >= NOW() - INTERVAL '1 month' ORDER BY created_at DESC",
    "explanation": "Fetches invoice documents from last month",
    "is_safe": true,
    "error": null
}}

User: "Who are my vendors?"
Output:
{{
    "sql": "SELECT DISTINCT vendor_name AS title, 'vendor' AS type, NULL AS amount, vendor_name AS vendor, NULL AS date FROM documents WHERE user_id = {user_id} AND vendor_name IS NOT NULL",
    "explanation": "Retrieves unique vendors",
    "is_safe": true,
    "error": null
}}

Strict Note:
- Always consider database schema while generating SQL
- Always use {user_id} placeholder for user filtering
- Don't help users with any other queries
- Don't generate SQL for any other purpose
- Don't helosinate or hallucinate
- For user details like: name, username etc. use the Users table

========================
 FINAL RULE:
========================
Every query MUST be UI-compatible and follow the exact column structure.
"""


        # Try GPT-4o first
        try:
            prompt = f"{system_prompt}\n\nConvert this query to SQL: {user_query}"
            response = self.openai_client.chat.completions.create(
                model=GPT4O_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048
            )
            
            text = response.choices[0].message.content
            # Clean up JSON if needed
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            
            # Additional safety checks
            sql = result.get("sql", "")
            
            # Must contain user filter (user_id for documents table, id for users table)
            has_user_filter = f"user_id = {user_id}" in sql or f"id = {user_id}" in sql
            if sql and not has_user_filter:
                return {
                    "sql": None,
                    "explanation": None,
                    "is_safe": False,
                    "error": f"Security error: Query must filter by user_id = {user_id} or id = {user_id}"
                }
            
            return result
        except Exception as e:
            logger.warning(f"GPT-4o SQL generation failed: {e}, trying Claude")
        
        # Fallback to Claude
        try:
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Convert this query to SQL: {user_query}"}
                ]
            )
            
            text = response.content[0].text if hasattr(response, 'content') else str(response)
            # Clean up JSON if needed
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            
            # Additional safety checks
            sql = result.get("sql", "")
            
            # Must contain user filter (user_id for documents table, id for users table)
            has_user_filter = f"user_id = {user_id}" in sql or f"id = {user_id}" in sql
            if sql and not has_user_filter:
                return {
                    "sql": None,
                    "explanation": None,
                    "is_safe": False,
                    "error": f"Security error: Query must filter by user_id = {user_id} or id = {user_id}"
                }
            
            return result
        except Exception as e:
            logger.error(f"Claude SQL generation also failed: {e}")
            return {
                "sql": None,
                "explanation": None,
                "is_safe": False,
                "error": f"Failed to generate SQL: {str(e)}"
            }

    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL against PostgreSQL (DATABASE_URL)."""
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

    def ask_ai(self, user_query: str, user_id: int) -> Dict[str, Any]:
        """
        Complete pipeline: Understand Intent -> Route -> Execute -> Format response.

        Args:
            user_query: Natural language question
            user_id: User's database ID for security filtering

        Returns:
            Complete response with SQL, results, and natural language answer
        """
        # Step 0: Understand what user actually wants
        intent_analysis = self._understand_intent(user_query)
        logger.info(f"Intent analysis for '{user_query}': {intent_analysis}")

        intent = intent_analysis.get("intent", "unknown")
        confidence = intent_analysis.get("confidence", 0)

        # Route based on intent
        if intent == "semantic_search" or intent == "unknown" or confidence < 0.6:
            # For topic-based queries or unclear intent, use semantic search directly
            logger.info(f"Routing to semantic search (intent: {intent}, confidence: {confidence})")
            return self._vector_search_fallback(user_query, user_id, f"Intent: {intent}, needs: {intent_analysis.get('needs')}")

        if intent == "greeting":
            return {
                "success": True,
                "sql": None,
                "explanation": None,
                "error": None,
                "data": None,
                "row_count": 0,
                "columns": None,
                "ai_response": "Hello! I'm your document assistant. I can help you find and analyze your documents. Ask me things like 'show my invoices' or 'tell me about my receipts'."
            }

        if intent == "conversation":
            return {
                "success": True,
                "sql": None,
                "explanation": None,
                "error": None,
                "data": None,
                "row_count": 0,
                "columns": None,
                "ai_response": "I'm a document management bot. I can help you search through your uploaded documents using natural language. Try asking about specific documents or topics!"
            }

        # Step 1: Generate SQL (only for sql_query and user_info intents)
        sql_result = self.generate_sql(user_query, user_id)

        # FALLBACK: If SQL generation fails, use vector search
        if not sql_result.get("is_safe") or not sql_result.get("sql"):
            logger.info(f"SQL generation failed for user {user_id}, falling back to vector search")
            return self._vector_search_fallback(user_query, user_id, sql_result.get("error"))

        sql = sql_result["sql"]

        # Step 2: Execute query
        exec_result = self.execute_query(sql)

        # FALLBACK: If SQL execution fails OR returns empty results, use vector search
        if exec_result.get("error"):
            logger.info(f"SQL execution failed for user {user_id}, falling back to vector search")
            return self._vector_search_fallback(user_query, user_id, exec_result["error"])

        # FALLBACK: If SQL returns empty results, try semantic search
        if exec_result.get("row_count", 0) == 0:
            logger.info(f"SQL returned 0 rows for user {user_id}, falling back to vector search")
            return self._vector_search_fallback(user_query, user_id, "SQL query returned no matching rows")

        # Step 3: Generate natural language response
        data = exec_result["rows"]
        ai_response = self._format_response(user_query, sql, data)

        # Step 4: Store SQL results in vector database for future semantic search
        # This allows the system to "remember" queried data
        try:
            self._store_sql_results(user_query, sql, data, user_id)
        except Exception as e:
            logger.warning(f"Failed to store SQL results in vector DB (non-critical): {e}")

        return {
            "success": True,
            "sql": sql,
            "explanation": sql_result.get("explanation"),
            "error": None,
            "data": data,
            "row_count": exec_result["row_count"],
            "columns": exec_result["columns"],
            "ai_response": ai_response
        }

    def _format_response(self, user_query: str, sql: str, data: List[Dict]) -> str:
        """Generate a natural language response from query results using LLM."""
        try:
            # Prepare context for LLM
            data_summary = json.dumps(data[:5], indent=2) if data else "[]"  # Limit to first 5 rows
            row_count = len(data)

            system_prompt = """You are a helpful data assistant. Summarize SQL query results in natural language.
Be concise but informative. Focus on the key insights the user asked for.
If there's an amount/currency, format it clearly.
Do not mention the SQL query itself, just the answer."""

            user_prompt = f"""Original question: "{user_query}"

Query results ({row_count} rows):
{data_summary}

Provide a natural language summary of these results that directly answers the user's question."""

            # Try GPT-4o first
            try:
                prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.openai_client.chat.completions.create(
                    model=GPT4O_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                )
                
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"GPT-4o response formatting failed: {e}, trying Claude")
            
            # Fallback to Claude
            try:
                response = self.claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                return response.content[0].text if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning(f"Claude response formatting also failed: {e}, falling back to raw data")
                
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, falling back to raw data")
        
        # Final fallback: return simple formatted data
        if not data:
            return "No results found."
        lines = []
        for i, row in enumerate(data[:5], 1):
            row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
            lines.append(f"{i}. {row_text}")
        return "\n".join(lines)

    def _store_sql_results(
        self,
        user_query: str,
        sql: str,
        data: List[Dict],
        user_id: int
    ) -> None:
        """
        Store SQL query and results in vector database for future semantic search.

        SECURITY: Always tags with user_id for guardrails.

        Args:
            user_query: Original natural language query
            sql: Generated SQL query
            data: Query results
            user_id: User ID (mandatory guardrail)
        """
        if not data:
            return

        try:
            # Create a searchable text representation of the query + results
            result_texts = []

            # Add the query context
            result_texts.append(f"Query: {user_query}")
            result_texts.append(f"SQL: {sql}")

            # Summarize results (limit to avoid huge embeddings)
            result_texts.append(f"Results ({len(data)} rows):")

            for i, row in enumerate(data[:5]):  # Store first 5 rows only
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                result_texts.append(f"  Row {i+1}: {row_text}")

            searchable_text = "\n".join(result_texts)

            # Generate unique ID for this query result
            import hashlib
            import time
            query_hash = hashlib.md5(f"{user_id}:{user_query}:{sql}".encode()).hexdigest()[:12]
            result_id = f"sql_result_{user_id}_{query_hash}_{int(time.time())}"

            # Generate embedding
            embedding = self._generate_embedding(searchable_text)

            # Store in ChromaDB with user_id guardrail
            self.vector_service.collection.upsert(
                ids=[result_id],
                embeddings=[embedding],
                metadatas=[{
                    "user_id": user_id,
                    "type": "sql_result",
                    "original_query": user_query,
                    "sql": sql[:500],  # Truncate for metadata
                    "row_count": len(data),
                    "text": searchable_text[:1000]
                }],
                documents=[searchable_text]
            )

            logger.info(f"Stored SQL results in vector DB: {result_id} for user {user_id}")

        except Exception as e:
            logger.error(f"Error storing SQL results in vector DB: {e}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        return response.data[0].embedding

    def _vector_search_fallback(
        self,
        user_query: str,
        user_id: int,
        error_reason: str = None
    ) -> Dict[str, Any]:
        """
        Semantic search fallback with simple result caching.
        Cache key: user_id + query_hash (5 min TTL)
        """
        import hashlib
        import time

        # Simple in-memory cache (user_id:query_hash -> (timestamp, results))
        cache_key = f"{user_id}:{hashlib.md5(user_query.lower().encode()).hexdigest()[:16]}"
        _CACHE_TTL_SECONDS = 300  # 5 minutes
        _MAX_CACHE_ENTRIES = 500

        # Check cache first
        if hasattr(self, '_search_cache') and cache_key in self._search_cache:
            cached_time, cached_result = self._search_cache[cache_key]
            if time.time() - cached_time < _CACHE_TTL_SECONDS:
                logger.debug(f"Cache hit for query: {user_query[:30]}...")
                cached_result['cached'] = True
                return cached_result

        try:
            # Perform semantic search with user_id guardrail
            results = self.vector_service.search(
                query=user_query,
                user_id=user_id,
                n_results=10
            )

            if not results:
                result = {
                    "success": False,
                    "sql": None,
                    "explanation": "Vector search fallback",
                    "error": error_reason or "No matching documents found",
                    "data": None,
                    "ai_response": "I am telegram bot i am not able to understand your query"
                }
                # Store in cache even for empty results
                if not hasattr(self, '_search_cache'):
                    self._search_cache = {}
                self._search_cache[cache_key] = (time.time(), result)
                return result

            # Format vector results as table data
            data = []
            for r in results:
                data.append({
                    "type": "document",
                    "title": f"Doc #{r['doc_id']} (Score: {r['similarity_score']}%)",
                    "amount": None,
                    "vendor": None,
                    "date": None,
                    "_text": r["text"],  # Internal field for reference
                    "_score": r["similarity_score"]
                })

            try:
                doc_ids = [int(r["doc_id"]) for r in results if r.get("doc_id") is not None]
                doc_details = DatabaseService.fetch_documents_for_vector_enrichment(user_id, doc_ids)

                for i, item in enumerate(data):
                    doc_id = results[i]["doc_id"]
                    try:
                        doc_id_int = int(doc_id)
                    except (TypeError, ValueError):
                        continue
                    if doc_id_int not in doc_details:
                        continue
                    d = doc_details[doc_id_int]
                    item.update({k: v for k, v in d.items() if v is not None})
                    item["type"] = d.get("document_type") or "document"
                    item["title"] = d.get("title") or f"Document {doc_id_int}"
                    item["amount"] = d.get("total_amount")
                    item["vendor"] = d.get("vendor_name")
                    item["date"] = d.get("created_at") or d.get("document_date")

            except Exception as e:
                logger.warning(f"Could not enrich vector results with DB data: {e}")

            # Filter results by minimum score threshold (50%)
            MIN_SCORE_THRESHOLD = 50.0
            filtered_data = [d for d in data if d.get('_score', 0) >= MIN_SCORE_THRESHOLD]
            
            # If no results pass threshold, show the best match with warning
            if not filtered_data and data:
                # Sort by score and take top 1 as best effort
                data_sorted = sorted(data, key=lambda x: x.get('_score', 0), reverse=True)
                filtered_data = data_sorted[:1]
                low_confidence = True
            else:
                low_confidence = False
                # Take top 3 best matches
                data_sorted = sorted(filtered_data, key=lambda x: x.get('_score', 0), reverse=True)
                filtered_data = data_sorted[:3]
            
            result = {
                "success": True,
                "sql": f"-- VECTOR SEARCH FALLBACK --\n-- Original query: {user_query}\n-- Guardrail: user_id = {user_id}",
                "explanation": f"Semantic search results (SQL fallback due to: {error_reason or 'unknown error'})",
                "error": None,
                "data": filtered_data,
                "row_count": len(filtered_data),
                "columns": ["type", "title", "amount", "vendor", "date"],
                "ai_response": "",  # Empty - UI will show table only
                "fallback": True,
                "fallback_reason": error_reason,
                "low_confidence": low_confidence
            }
            # Store successful result in cache
            if not hasattr(self, '_search_cache'):
                self._search_cache = {}
            self._search_cache[cache_key] = (time.time(), result)
            # Cleanup old cache entries
            if len(self._search_cache) > _MAX_CACHE_ENTRIES:
                oldest = sorted(self._search_cache.items(), key=lambda x: x[1][0])[:100]
                for k, _ in oldest:
                    del self._search_cache[k]
            return result

        except Exception as e:
            logger.error(f"Vector search fallback failed: {e}")
            return {
                "success": False,
                "sql": None,
                "explanation": "Both SQL and vector search failed",
                "error": f"SQL failed: {error_reason}. Vector search also failed: {str(e)}",
                "data": None,
                "ai_response": f"I couldn't process your query. Please try rephrasing it."
            }


# Singleton instance
_nlp_sql_service = None


def get_nlp_sql_service() -> NLPSQLService:
    """Get or create the NLP SQL service singleton."""
    global _nlp_sql_service
    if _nlp_sql_service is None:
        _nlp_sql_service = NLPSQLService()
    return _nlp_sql_service
