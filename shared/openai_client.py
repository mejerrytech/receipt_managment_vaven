import os
import asyncio
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import anthropic

load_dotenv()

logger = logging.getLogger("openai_service")

DEFAULT_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "45"))
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
MAX_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

# Model configuration
GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-opus-4-5-20251101"


class OpenAIService:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        # Initialize Gemini client (primary)
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        # Initialize Claude client (fallback)
        self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.max_retries = max_retries
        self.conversations: Dict[str, List[Dict]] = {}

    def _resp_with_gemini(self, prompt: str, system_prompt: str | None = None, use_search: bool = False) -> Optional[str]:
        """Generate response using Gemini 2.5 Flash."""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
            
            # Add system prompt if provided (Gemini doesn't have separate system field, prepend to prompt)
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
                contents[0].parts = [types.Part.from_text(text=prompt)]
            
            # Note: Gemini doesn't have built-in web_search tool like OpenAI
            # For web search, we'd need to implement it separately or use a different approach
            if use_search:
                logger.warning("Web search requested but Gemini doesn't have built-in web search tool")
            
            response = self.gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                )
            )
            
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Gemini response failed: {e}")
            return None

    def _resp_with_claude(self, prompt: str, system_prompt: str | None = None, use_search: bool = False) -> Optional[str]:
        """Generate response using Claude Opus 4.5 (fallback)."""
        try:
            # Note: Claude doesn't have built-in web search either
            if use_search:
                logger.warning("Web search requested but Claude doesn't have built-in web search tool")
            
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0.7,
                system=system_prompt or "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Claude response failed: {e}")
            return None

    def _resp(self, prompt: str, system_prompt: str | None = None, use_search: bool = False) -> str:
        """Try Gemini first, fallback to Claude if it fails."""
        # Try Gemini first
        result = self._resp_with_gemini(prompt, system_prompt, use_search)
        if result:
            return result
        
        # Fallback to Claude
        logger.info("Gemini failed, falling back to Claude")
        result = self._resp_with_claude(prompt, system_prompt, use_search)
        if result:
            return result
        
        # Both failed
        return "Sorry — I had trouble processing that request."

    async def _resp_async(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self._resp, *args, **kwargs)

    async def ask(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        chat_id: str | None = None,
        use_memory: bool = True,
    ) -> str:
        if use_memory and chat_id and chat_id in self.conversations:
            history = self.conversations[chat_id][-MAX_HISTORY:]
            hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
            user_prompt = f"Conversation so far:\n{hist_text}\n\nUser: {user_prompt}"

        for attempt in range(self.max_retries + 1):
            try:
                answer = await self._resp_async(
                    user_prompt, system_prompt=system_prompt, use_search=False
                )
                if use_memory and chat_id:
                    self.conversations.setdefault(chat_id, []).extend([
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": answer},
                    ])
                    # Store 2x MAX_HISTORY (user + assistant pairs)
                    self.conversations[chat_id] = self.conversations[chat_id][-(MAX_HISTORY * 2):]
                return answer
            except (APITimeoutError, APIError, Exception) as e:
                logger.error(f"OpenAI ask failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
        return "Sorry — I had trouble processing that request."

    async def web_search(
        self, user_prompt: str, system_prompt: str | None = None
    ) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                return await self._resp_async(
                    user_prompt, system_prompt=system_prompt, use_search=True
                )
            except (APITimeoutError, APIError, Exception) as e:
                logger.error(f"Web search failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
        # Fallback to regular ask without search
        return await self.ask(user_prompt, system_prompt=system_prompt, use_memory=False)

    def clear_conversation(self, chat_id: str) -> None:
        if chat_id in self.conversations:
            del self.conversations[chat_id]
            logger.info(f"Cleared conversation history for chat {chat_id}")
