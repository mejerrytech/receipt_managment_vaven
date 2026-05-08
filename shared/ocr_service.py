"""
Multi-model OCR Service with confidence-based routing.

Primary: GPT-4o (Vision)
Fallback: Claude Opus 4.5

Flow:
1. Send image to GPT-4o Vision first
2. Parse JSON response, compute confidence scores
3. If all critical fields ≥ 0.80 → accept result
4. If any critical field < 0.60 OR JSON parse fails → retry with Claude Opus 4.5
5. Take the better result of the two
"""

import os
import json
import asyncio
import logging
import base64
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Import OpenAI for GPT-4o
import openai

# Import Anthropic for Claude
import anthropic

load_dotenv()

logger = logging.getLogger("ocr_service")

# Model configuration
GPT4O_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-opus-4-5-20251101"  # Claude Opus 4.5

# Confidence thresholds
CONFIRMED_THRESHOLD = 0.80  # All critical fields ≥ this = confirmed
FALLBACK_THRESHOLD = 0.60   # Any critical field < this = try Claude


@dataclass
class OCRResult:
    """OCR extraction result with metadata."""
    data: Dict[str, Any]
    confidence: Dict[str, float]
    overall_confidence: float
    model_used: str
    status: str  # 'confirmed' or 'pending_review'
    fallback_used: bool = False


class OCRService:
    """Multi-model OCR service with confidence-based routing."""
    
    def __init__(self):
        # Initialize OpenAI client (GPT-4o)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.error("OPENAI_API_KEY not set!")
        self.openai_client = openai.OpenAI(api_key=openai_key)
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Default system prompt for OCR
        self.system_prompt = """You are an advanced document and image data extraction engine with strong OCR and deep reasoning capabilities.

Carefully analyze the provided image or document, perform accurate OCR to read all visible text, and use deep contextual understanding to extract all meaningful and structured information.

Infer relationships, correct minor OCR errors, and normalize the data where appropriate.

Return ONLY valid JSON with no markdown, no backticks, no preamble, and no explanation. If a field cannot be determined, use null.

Include confidence scores for each extracted field (0.0 to 1.0)."""

        # Default extraction prompt
        self.extraction_prompt = """Extract all visible data from this document/image.

Return a JSON object with these fields:
{
  "document_type": "string or null - what type of document this is",
  "title": "string or null",
  "date": "YYYY-MM-DD or null",
  "amounts": {"total": number_or_null, "subtotal": number_or_null, "tax": number_or_null, "currency": "INR or detected currency"},
  "vendor_or_sender": {"name": "string or null", "address": "string or null", "contact": "string or null"},
  "recipient": {"name": "string or null", "address": "string or null"},
  "identifiers": {"invoice_number": "string or null", "order_id": "string or null", "gstin": "string or null", "other_ids": []},
  "items": [{"description": "string", "quantity": number_or_null, "price": number_or_null, "amount": number_or_null}],
  "text_content": "string or null - any important text extracted from the document",
  "tables": [],
  "confidence": {
    "overall": 0.0_to_1.0,
    "document_type": 0.0_to_1.0,
    "title": 0.0_to_1.0,
    "date": 0.0_to_1.0,
    "amounts_total": 0.0_to_1.0,
    "vendor_name": 0.0_to_1.0,
    "invoice_number": 0.0_to_1.0
  }
}

Critical fields for confidence assessment:
- amounts.total (financial accuracy critical)
- vendor_or_sender.name (vendor identification critical)
- document_type (classification critical)
- identifiers.invoice_number (document identification critical)
- title (document identification important)
- date (financial/temporal accuracy important)

Adapt the fields based on the document type. For non-financial documents, include relevant fields."""

        self.expense_categories = [
            "Food and Dining", "Groceries", "Rent", "Utilities", "Fual", "Shopping",
            "Entertainment", "Healthcare", "Edication", "Personal care", "Subscription",
            "EMI/Loans", "Insurance", "Investment", "Travel", "Savings", "CAB/Taxi",
            "Misecellaneous", "Other"
        ]

    def _build_effective_prompt(
        self,
        custom_prompt: Optional[str] = None,
        user_input_text: Optional[str] = None
    ) -> str:
        base_prompt = custom_prompt or self.extraction_prompt
        categories_text = ", ".join(self.expense_categories)
        user_text_section = (
            f'\n\nUser-provided upload text/intention:\n"{user_input_text}"\n'
            "Use this text as additional context while extracting fields."
            if user_input_text else
            "\n\nUser-provided upload text/intention: null"
        )
        return (
            f"{base_prompt}\n\n"
            "Also infer a best-fit `expense_category` from this list (if possible):\n"
            f"{categories_text}\n"
            "If not inferable, set expense_category to \"Other\"."
            f"{user_text_section}"
        )

    def _encode_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Encode image to base64."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _parse_confidence_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence scores from OCR result."""
        confidence = data.get("confidence", {})
        
        # Extract individual field confidences
        scores = {
            "overall": confidence.get("overall", 0.0) if isinstance(confidence, dict) else 0.0,
            "document_type": confidence.get("document_type", 0.0) if isinstance(confidence, dict) else 0.0,
            "title": confidence.get("title", 0.0) if isinstance(confidence, dict) else 0.0,
            "date": confidence.get("date", 0.0) if isinstance(confidence, dict) else 0.0,
            "amounts_total": confidence.get("amounts_total", 0.0) if isinstance(confidence, dict) else 0.0,
            "vendor_name": confidence.get("vendor_name", 0.0) if isinstance(confidence, dict) else 0.0,
            "invoice_number": confidence.get("invoice_number", 0.0) if isinstance(confidence, dict) else 0.0,
        }
        
        return scores
    
    def _assess_quality(self, confidence_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Assess OCR quality based on confidence scores.
        
        Returns:
            Tuple of (status, min_critical_score)
            status: 'confirmed' or 'pending_review'
        """
        # Critical fields for financial documents
        critical_fields = ["amounts_total", "vendor_name", "document_type", "invoice_number"]
        
        critical_scores = [
            confidence_scores.get(field, 0.0) 
            for field in critical_fields
        ]
        
        min_critical = min(critical_scores) if critical_scores else 0.0
        
        # Determine status
        if min_critical >= CONFIRMED_THRESHOLD:
            return "confirmed", min_critical
        else:
            return "pending_review", min_critical
    
    async def _extract_with_gpt4o(
        self,
        image_bytes: bytes,
        mime_type: str,
        custom_prompt: Optional[str] = None,
        user_input_text: Optional[str] = None
    ) -> Optional[OCRResult]:
        """Extract data using GPT-4o Vision."""
        try:
            user_prompt = self._build_effective_prompt(custom_prompt, user_input_text)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=GPT4O_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=4096
            )

            # Parse JSON response
            text = response.choices[0].message.content

            # Clean up JSON (remove markdown code blocks if present)
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)

            # Parse confidence scores
            confidence = self._parse_confidence_scores(data)

            # Assess quality
            status, min_critical = self._assess_quality(confidence)

            return OCRResult(
                data=data,
                confidence=confidence,
                overall_confidence=confidence["overall"],
                model_used="gpt-4o",
                status=status,
                fallback_used=False
            )

        except Exception as e:
            logger.error(f"GPT-4o OCR failed: {e}")
            return None
    
    async def _extract_with_claude(
        self, 
        image_bytes: bytes, 
        mime_type: str,
        custom_prompt: Optional[str] = None,
        user_input_text: Optional[str] = None
    ) -> Optional[OCRResult]:
        """Extract data using Claude Opus 4.5."""
        try:
            base64_image = self._encode_image(image_bytes, mime_type)
            user_prompt = self._build_effective_prompt(custom_prompt, user_input_text)
            
            response = await asyncio.to_thread(
                self.claude_client.messages.create,
                model=CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0.1,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Parse JSON response
            text = response.content[0].text if hasattr(response, 'content') else str(response)
            
            # Clean up JSON
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            
            # Parse confidence scores
            confidence = self._parse_confidence_scores(data)
            
            # Assess quality
            status, min_critical = self._assess_quality(confidence)
            
            return OCRResult(
                data=data,
                confidence=confidence,
                overall_confidence=confidence["overall"],
                model_used="claude-sonnet-4.6",
                status=status,
                fallback_used=True
            )
            
        except Exception as e:
            logger.error(f"Claude OCR failed: {e}")
            return None
    
    async def extract_data(
        self, 
        image_bytes: bytes, 
        mime_type: str,
        custom_prompt: Optional[str] = None,
        user_input_text: Optional[str] = None
    ) -> str:
        """
        Extract data from image with confidence-based model routing.
        
        Flow:
        1. Try GPT-4o Vision first
        2. If confidence < 0.60 or JSON parse fails → try Claude Opus 4.5
        3. Return the better result
        
        Returns:
            JSON string with extracted data and metadata
        """
        logger.info("Starting OCR extraction with GPT-4o Vision (primary)")

        # Step 1: Try GPT-4o first
        gpt4o_result = await self._extract_with_gpt4o(image_bytes, mime_type, custom_prompt, user_input_text)

        if gpt4o_result is None:
            logger.warning("GPT-4o OCR failed, falling back to Claude Opus 4.5")
            claude_result = await self._extract_with_claude(image_bytes, mime_type, custom_prompt, user_input_text)
            
            if claude_result is None:
                return json.dumps({
                    "error": "Both primary and fallback OCR models failed",
                    "status": "failed"
                })
            
            return json.dumps({
                **claude_result.data,
                "_ocr_metadata": {
                    "model_used": claude_result.model_used,
                    "overall_confidence": claude_result.overall_confidence,
                    "field_confidences": claude_result.confidence,
                    "status": claude_result.status,
                    "fallback_used": True
                }
            })

        # Step 2: Check if we need fallback
        min_critical = min([
            gpt4o_result.confidence.get("amounts_total", 0.0),
            gpt4o_result.confidence.get("vendor_name", 0.0),
            gpt4o_result.confidence.get("document_type", 0.0),
            gpt4o_result.confidence.get("invoice_number", 0.0)
        ]) if gpt4o_result.confidence else 0.0

        if min_critical < FALLBACK_THRESHOLD:
            logger.warning(f"GPT-4o confidence too low ({min_critical:.2f}), trying Claude Opus 4.5")
            
            claude_result = await self._extract_with_claude(image_bytes, mime_type, custom_prompt, user_input_text)
            
            if claude_result:
                # Compare and take the better result
                claude_min_critical = min([
                    claude_result.confidence.get("amounts_total", 0.0),
                    claude_result.confidence.get("vendor_name", 0.0),
                    claude_result.confidence.get("document_type", 0.0),
                    claude_result.confidence.get("invoice_number", 0.0)
                ])
                
                logger.info(f"GPT-4o min critical: {min_critical:.2f}, Claude min critical: {claude_min_critical:.2f}")

                # Use the result with higher critical field confidence
                if claude_min_critical > min_critical:
                    logger.info("Claude result is better, using Claude")
                    return json.dumps({
                        **claude_result.data,
                        "_ocr_metadata": {
                            "model_used": claude_result.model_used,
                            "overall_confidence": claude_result.overall_confidence,
                            "field_confidences": claude_result.confidence,
                            "status": claude_result.status,
                            "fallback_used": True,
                            "comparison": f"Claude better ({claude_min_critical:.2f} > {min_critical:.2f})"
                        }
                    })
                else:
                    logger.info("GPT-4o result is better or equal, keeping GPT-4o")

        # Return GPT-4o result (either good enough or better than Claude)
        return json.dumps({
            **gpt4o_result.data,
            "_ocr_metadata": {
                "model_used": gpt4o_result.model_used,
                "overall_confidence": gpt4o_result.overall_confidence,
                "field_confidences": gpt4o_result.confidence,
                "status": gpt4o_result.status,
                "fallback_used": gpt4o_result.fallback_used
            }
        })


# Global instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create global OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
