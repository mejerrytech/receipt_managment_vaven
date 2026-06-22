from typing import Optional

from pydantic import BaseModel


class ReceiptUpdateRequest(BaseModel):
    vendor_name: Optional[str] = None
    gstin: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_number: Optional[str] = None
    total_amount: Optional[float] = None
    gst_amount: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None
