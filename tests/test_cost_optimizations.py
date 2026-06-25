import unittest

from shared.message_heuristics import (
    greeting_reply,
    is_simple_greeting,
    looks_like_expense_message,
    looks_like_multi_item_expense,
)
from shared.receipt_card_formatter import (
    build_whatsapp_review_from_ocr,
    build_whatsapp_review_messages,
    format_receipt_card_markdown,
)


class MessageHeuristicsTests(unittest.TestCase):
    def test_greeting_detected(self):
        self.assertTrue(is_simple_greeting("Hiii"))
        self.assertTrue(is_simple_greeting("  hello!  "))
        self.assertIsNotNone(greeting_reply("thanks"))

    def test_expense_vs_question(self):
        self.assertTrue(looks_like_expense_message("Maine 500 ka petrol dalwaya"))
        self.assertFalse(looks_like_expense_message("show my invoices"))
        self.assertFalse(looks_like_expense_message("Hiii"))

    def test_multi_item(self):
        self.assertTrue(
            looks_like_multi_item_expense(
                "Maine 2000 ka petrol dalwaya. Maine 66 ka milk aur 140 ka tel liya."
            )
        )
        self.assertFalse(looks_like_multi_item_expense("Hiii"))


class ReceiptCardFormatterTests(unittest.TestCase):
    def test_formats_card_and_whatsapp_footer(self):
        ocr = {
            "document_type": "Tax Invoice",
            "vendor_name": "ABC Store",
            "document_date": "2026-01-01",
            "total_amount": 722.4,
            "items": [
                {"description": "Milk", "quantity": 1, "amount": 66},
                {"description": "Oil", "quantity": 1, "amount": 140},
            ],
        }
        card = format_receipt_card_markdown(ocr)
        self.assertIn("🏪 Vendor: ABC Store", card)
        self.assertIn("ITEMS (2)", card)

        review = build_whatsapp_review_messages(ocr, pending_id=23, max_chars=1500)
        self.assertEqual(len(review), 1)
        self.assertIn("Pending ID: 23", review[0])
        self.assertIn("CONFIRM 23", review[0])
        self.assertLessEqual(len(review[0]), 1500)

    def test_many_items_stay_in_one_whatsapp_card(self):
        items = [
            {"Particulars": f"Product {i}", "Qty/Kg": 1, "N/Rate": i * 10, "Value": i * 10}
            for i in range(1, 37)
        ]
        ocr = {
            "vendor_name": "Big Bazaar",
            "document_date": "2026-01-01",
            "subtotal": 6660,
            "cgst": 333,
            "sgst": 333,
            "total_amount": 7326,
            "items": items,
        }
        messages = build_whatsapp_review_messages(ocr, pending_id=30)
        self.assertEqual(len(messages), 1)
        self.assertLessEqual(len(messages[0]), 1590)
        self.assertIn("ITEMS (36)", messages[0])
        self.assertIn("Product 1", messages[0])
        self.assertIn("Product 36", messages[0])
        self.assertIn("CONFIRM 30", messages[0])

    def test_indian_invoice_column_names(self):
        """Gemini often returns Particular / Qty/Kg / Rate / Value on Indian invoices."""
        ocr = {
            "vendor_name": "SRT",
            "document_date": "02/02/2025",
            "total_amount": 4086.92,
            "items": [
                {"Particular": "L WADA", "Qty/Kg": 2.016, "Rate": 64.0, "Value": 129.15},
                {"Particular": "PURO H-1kg", "Qty/Kg": 1, "Rate": 85.0, "Value": 85.0},
            ],
        }
        card = format_receipt_card_markdown(ocr)
        self.assertIn("ITEMS (2)", card)
        self.assertIn("L WADA", card)
        self.assertIn("PURO H-1kg", card)
        self.assertIn("₹129.15", card)

    def test_dmart_particulars_and_n_rate(self):
        ocr = {
            "vendor_name": "D Mart",
            "total_amount": 4010.04,
            "items": [
                {"Particulars": "BRITANNIA CHEE-400g", "Qty/Kg": 1, "N/Rate": 216.0, "Value": 216.0},
            ],
        }
        card = format_receipt_card_markdown(ocr)
        self.assertIn("BRITANNIA CHEE-400g", card)
        self.assertIn("1 x ₹216.00", card)


class GeminiOCRParseTests(unittest.TestCase):
    def test_salvage_truncated_json(self):
        from shared.gemini_ocr_service import GeminiOCRService

        raw = """```json
{
  "status": "readable",
  "document_type": "Retail Receipt",
  "currency": "INR",
  "total_amount": 2686.31,
  "subtotal": 2525.95,
  "tax_details": {
    "cgst_total": 80.18,
    "tax_slabs":
```"""
        service = GeminiOCRService.__new__(GeminiOCRService)
        salvaged = service._salvage_truncated_json(raw)
        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged.get("total_amount"), 2686.31)


if __name__ == "__main__":
    unittest.main()
