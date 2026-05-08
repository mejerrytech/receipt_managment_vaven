"""
Orchestration Layer - AI System Control Center

This module manages:
- Model selection (GPT-4o primary, Anthropic fallback)
- Function calling orchestration
- Execution order and pipeline management
- Error handling and fallback logic
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Type
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import anthropic

load_dotenv()

logger = logging.getLogger("orchestrator")

# Model configuration
GPT4O_MODEL = "gpt-4o"
GPT4O_MINI = "gpt-4o-mini"
CLAUDE_MODEL = "claude-opus-4-5-20251101"
CLAUDE_SONNET = "claude-sonnet-4-5-20251101"


class AgentType(Enum):
    """Types of agents in the system."""
    NLP_TO_SQL = "nlp_to_sql"
    RAG = "rag"
    INTENT_CLASSIFIER = "intent_classifier"
    RESPONSE_FORMATTER = "response_formatter"


class ModelProvider(Enum):
    """Available model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class AgentStep:
    """Represents a single step in the agent pipeline."""
    name: str
    agent_type: AgentType
    function_schemas: List[Dict[str, Any]]
    execute_fn: Callable
    fallback_provider: ModelProvider = ModelProvider.ANTHROPIC
    max_retries: int = 2


@dataclass
class ExecutionResult:
    """Result of an agent step execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    provider_used: ModelProvider = ModelProvider.OPENAI
    function_calls: List[Dict] = None


class ModelOrchestrator:
    """
    Orchestration layer - Control center for AI system.

    Responsibilities:
    1. Select appropriate model (GPT-4o primary, Anthropic fallback)
    2. Execute function calling for agents
    3. Manage step execution order
    4. Handle errors and fallbacks
    """

    def __init__(self):
        # Initialize OpenAI client (GPT-4o - primary)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not set! GPT-4o will not work.")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

        # Initialize Anthropic client (fallback)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY not set! Fallback will not work.")
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Track execution history for debugging
        self.execution_history: List[Dict] = []

        logger.info("ModelOrchestrator initialized with GPT-4o primary and Anthropic fallback")

    def _call_gpt4o_with_functions(
        self,
        system_prompt: str,
        user_message: str,
        functions: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Optional[Dict[str, Any]]:
        """
        Call GPT-4o with function calling support.

        Args:
            system_prompt: System instructions
            user_message: User query
            functions: Function schemas for function calling
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Response dict with function_call or content
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = self.openai_client.chat.completions.create(
                model=GPT4O_MODEL,
                messages=messages,
                functions=functions if functions else None,
                function_call="auto" if functions else None,
                temperature=temperature,
                max_tokens=max_tokens
            )

            message = response.choices[0].message

            result = {
                "content": message.content,
                "function_call": None,
                "provider": ModelProvider.OPENAI,
                "model": GPT4O_MODEL
            }

            # Check if function was called
            if message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments
                }
                logger.info(f"GPT-4o called function: {message.function_call.name}")

            return result

        except Exception as e:
            logger.error(f"GPT-4o call failed: {e}")
            return None

    def _call_claude_fallback(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude as fallback when GPT-4o fails.

        Note: Claude doesn't support function calling in the same way,
        so we convert function schemas to instructions in the prompt.
        """
        try:
            response = self.anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            content = response.content[0].text if hasattr(response, 'content') else str(response)

            return {
                "content": content,
                "function_call": None,
                "provider": ModelProvider.ANTHROPIC,
                "model": CLAUDE_MODEL
            }

        except Exception as e:
            logger.error(f"Claude fallback failed: {e}")
            return None

    def execute_with_fallback(
        self,
        system_prompt: str,
        user_message: str,
        functions: List[Dict[str, Any]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_retries: int = 2
    ) -> ExecutionResult:
        """
        Execute with GPT-4o primary, fallback to Anthropic if fails.

        This is the core orchestration method that handles:
        - Primary model selection (GPT-4o)
        - Function calling
        - Error handling and retries
        - Fallback to secondary model (Claude)

        Args:
            system_prompt: System instructions
            user_message: User query
            functions: Function schemas for function calling
            temperature: Temperature for generation
            max_tokens: Max tokens to generate
            max_retries: Number of retries for primary model

        Returns:
            ExecutionResult with success status and data
        """
        # Step 1: Try GPT-4o with function calling (primary)
        for attempt in range(max_retries + 1):
            result = self._call_gpt4o_with_functions(
                system_prompt=system_prompt,
                user_message=user_message,
                functions=functions,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if result:
                # Log successful execution
                self.execution_history.append({
                    "provider": "openai",
                    "model": GPT4O_MODEL,
                    "success": True,
                    "attempt": attempt + 1,
                    "has_function_call": result.get("function_call") is not None
                })

                function_calls = []
                if result.get("function_call"):
                    function_calls = [result["function_call"]]

                return ExecutionResult(
                    success=True,
                    data=result,
                    provider_used=ModelProvider.OPENAI,
                    function_calls=function_calls
                )

            logger.warning(f"GPT-4o attempt {attempt + 1} failed, retrying...")

        # Step 2: Fallback to Claude if all GPT-4o attempts failed
        logger.info("All GPT-4o attempts failed, falling back to Anthropic Claude")

        # For Claude fallback, we need to adjust the prompt since it doesn't
        # support function calling the same way
        adjusted_prompt = system_prompt
        if functions:
            adjusted_prompt += f"\n\nAvailable functions:\n{json.dumps(functions, indent=2)}"
            adjusted_prompt += "\n\nIf you need to call a function, respond with JSON: {\"function_call\": {\"name\": \"...\", \"arguments\": {...}}}"

        result = self._call_claude_fallback(
            system_prompt=adjusted_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if result:
            self.execution_history.append({
                "provider": "anthropic",
                "model": CLAUDE_MODEL,
                "success": True,
                "fallback": True
            })

            # Try to parse function call from Claude response
            function_calls = self._extract_function_calls_from_claude(result.get("content", ""))

            return ExecutionResult(
                success=True,
                data=result,
                provider_used=ModelProvider.ANTHROPIC,
                function_calls=function_calls if function_calls else None
            )

        # Both failed
        self.execution_history.append({
            "provider": "both",
            "success": False,
            "error": "Both primary and fallback models failed"
        })

        return ExecutionResult(
            success=False,
            data=None,
            error="Both GPT-4o and Claude failed to respond",
            provider_used=ModelProvider.OPENAI
        )

    def _extract_function_calls_from_claude(self, content: str) -> List[Dict]:
        """
        Extract function calls from Claude's text response.
        Claude doesn't have native function calling, so we parse JSON from text.
        """
        function_calls = []
        try:
            # Look for JSON function call patterns
            if "function_call" in content:
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    json_str = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    json_str = content[start:end].strip()
                else:
                    # Try to find JSON directly
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end].strip()

                parsed = json.loads(json_str)
                if "function_call" in parsed:
                    function_calls.append(parsed["function_call"])
        except Exception as e:
            logger.debug(f"Could not extract function calls from Claude response: {e}")

        return function_calls

    def execute_step(
        self,
        step: AgentStep,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a single agent step in the pipeline.

        Args:
            step: The step to execute
            context: Context data for the step (user_id, query, etc.)

        Returns:
            ExecutionResult
        """
        logger.info(f"Executing step: {step.name} ({step.agent_type.value})")

        # Build system prompt based on agent type
        system_prompt = self._build_system_prompt(step.agent_type, context)

        # Build user message
        user_message = self._build_user_message(step.agent_type, context)

        # Execute with orchestration
        result = self.execute_with_fallback(
            system_prompt=system_prompt,
            user_message=user_message,
            functions=step.function_schemas,
            max_retries=step.max_retries
        )

        # If function calls were made, execute them
        if result.success and result.function_calls:
            for func_call in result.function_calls:
                func_name = func_call.get("name")
                arguments = func_call.get("arguments")

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                logger.info(f"Executing function: {func_name} with args: {arguments}")

                # Call the step's execute function with the function call
                try:
                    func_result = step.execute_fn(func_name, arguments, context)
                    # Update context with function result
                    context[f"{func_name}_result"] = func_result
                except Exception as e:
                    logger.error(f"Function execution failed: {e}")
                    return ExecutionResult(
                        success=False,
                        data=None,
                        error=f"Function {func_name} execution failed: {str(e)}",
                        provider_used=result.provider_used
                    )

        return result

    def _build_system_prompt(self, agent_type: AgentType, context: Dict) -> str:
        """Build appropriate system prompt for each agent type."""
        if agent_type == AgentType.NLP_TO_SQL:
            return self._get_nlp_to_sql_system_prompt(context)
        elif agent_type == AgentType.RAG:
            return self._get_rag_system_prompt(context)
        elif agent_type == AgentType.INTENT_CLASSIFIER:
            return self._get_intent_classifier_prompt()
        elif agent_type == AgentType.RESPONSE_FORMATTER:
            return self._get_response_formatter_prompt()
        return "You are a helpful assistant."

    def _build_user_message(self, agent_type: AgentType, context: Dict) -> str:
        """Build user message for each agent type."""
        if agent_type == AgentType.NLP_TO_SQL:
            return context.get("user_query", "")
        elif agent_type == AgentType.RAG:
            return f"Query: {context.get('user_query', '')}\nDocuments: {context.get('retrieved_docs', [])}"
        elif agent_type == AgentType.INTENT_CLASSIFIER:
            return f"Classify this query: {context.get('user_query', '')}"
        elif agent_type == AgentType.RESPONSE_FORMATTER:
            return f"Format this data: {context.get('data', {})}"
        return ""

    def _get_nlp_to_sql_system_prompt(self, context: Dict) -> str:
        """Get system prompt for NLP to SQL agent."""
        user_id = context.get("user_id", 0)
        return f"""You are an AI assistant that converts natural language questions into safe SQLite SQL queries.

DATABASE SCHEMA:
Tables:
1. users
   - id (INTEGER PRIMARY KEY)
   - telegram_id (BIGINT UNIQUE)
   - first_name (VARCHAR)
   - last_name (VARCHAR)
   - username (VARCHAR)
   - created_at (DATETIME)

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

CRITICAL SECURITY RULES:
1. ALWAYS include "WHERE user_id = {user_id}" in the query (MANDATORY)
2. NEVER access data of other users
3. ONLY generate SELECT queries (NO INSERT, UPDATE, DELETE, DROP, ALTER)
4. "my" always refers to user_id = {user_id}
5. Return only valid SQLite SQL

UI RESPONSE FORMAT RULES:
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

Respond with JSON in this format:
{{
    "sql": "SELECT ...",
    "explanation": "What this query does",
    "is_safe": true,
    "error": null
}}"""

    def _get_rag_system_prompt(self, context: Dict) -> str:
        """Get system prompt for RAG agent."""
        return """You are a document analysis assistant. Answer questions based on the provided documents.

Use the available functions to search documents and extract relevant information.
Be concise and accurate. Cite specific document details when possible.

If you need to search for more documents, use the search_documents function.
If you need to get document details, use the get_document_details function."""

    def _get_intent_classifier_prompt(self) -> str:
        """Get system prompt for intent classifier."""
        return """You are an intent classifier for a document management bot.

Analyze the user's query and classify the intent. Respond in JSON format.

Intent Types:
- "sql_query": User wants specific data from their documents (show invoices, find receipts, totals, etc.)
- "semantic_search": User asks about content/topics in documents (tell me about AI, what do I have about X, etc.)
- "conversation": General chat, questions about the bot, help, etc.
- "greeting": Hello, hi, thanks, bye, etc.
- "user_info": User asks about their profile, name, settings
- "unknown": Unclear what user wants

Respond ONLY with valid JSON:
{
    "intent": "sql_query|semantic_search|conversation|greeting|user_info|unknown",
    "needs": "description of what user wants",
    "confidence": 0.0-1.0,
    "suggested_action": "generate_sql|vector_fallback|chat_response|greet|explain_bot"
}"""

    def _get_response_formatter_prompt(self) -> str:
        """Get system prompt for response formatter."""
        return """You are a helpful assistant. Answer the user's question based on the data found.

Rules:
1. Answer ONLY what was asked - no extra information
2. Be conversational and human-like (natural, friendly tone)
3. Format amounts/currency clearly
4. Do not mention SQL or technical details
5. Keep it brief but complete

Examples:
- User: "maatha agencies ka kitna bill h" → "MAATHA AGENCIES ka bill ₹7222.40 hai"
- User: "show my invoices" → List only the invoices, no extra explanation"""

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution history for debugging."""
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get("success"))
        openai_count = sum(1 for h in self.execution_history if h.get("provider") == "openai")
        anthropic_count = sum(1 for h in self.execution_history if h.get("provider") == "anthropic")
        fallback_count = sum(1 for h in self.execution_history if h.get("fallback"))

        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "openai_primary": openai_count,
            "anthropic_fallback": anthropic_count,
            "fallback_rate": fallback_count / total if total > 0 else 0
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all model providers.
        Tests both GPT-4o (primary) and Anthropic (fallback).
        """
        results = {
            "openai": {"status": "unknown", "latency_ms": None, "error": None},
            "anthropic": {"status": "unknown", "latency_ms": None, "error": None},
            "overall": "unknown"
        }

        import time

        # Test OpenAI (GPT-4o)
        try:
            start = time.time()
            response = self.openai_client.chat.completions.create(
                model=GPT4O_MINI,  # Use mini for faster health check
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10
            )
            latency = (time.time() - start) * 1000
            results["openai"] = {
                "status": "healthy" if response.choices[0].message.content else "degraded",
                "latency_ms": round(latency, 2),
                "error": None
            }
        except Exception as e:
            results["openai"] = {"status": "unhealthy", "latency_ms": None, "error": str(e)}

        # Test Anthropic (Claude)
        try:
            start = time.time()
            response = self.anthropic_client.messages.create(
                model=CLAUDE_SONNET,  # Use sonnet for faster health check
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'OK'"}]
            )
            latency = (time.time() - start) * 1000
            results["anthropic"] = {
                "status": "healthy" if response.content else "degraded",
                "latency_ms": round(latency, 2),
                "error": None
            }
        except Exception as e:
            results["anthropic"] = {"status": "unhealthy", "latency_ms": None, "error": str(e)}

        # Determine overall status
        if results["openai"]["status"] == "healthy":
            results["overall"] = "healthy"  # Primary is good
        elif results["anthropic"]["status"] == "healthy":
            results["overall"] = "degraded"  # Fallback available
        else:
            results["overall"] = "unhealthy"  # Both down

        return results


# Singleton instance
_orchestrator = None


def get_orchestrator() -> ModelOrchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ModelOrchestrator()
    return _orchestrator
