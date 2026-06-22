from typing import Any, Optional

from pydantic import BaseModel


class ReportPreviewResponse(BaseModel):
    period_label: str
    start_date: str
    end_date: str
    total_invoices: int
    total_amount: float
    total_gst: float
    itc_eligible: float
    currency: str


class ReportFileLinks(BaseModel):
    pdf: Optional[str] = None
    csv: Optional[str] = None


class GenerateReportResponse(BaseModel):
    success: bool
    report_id: str
    format: str
    files: ReportFileLinks
    preview: ReportPreviewResponse
    message: str


class ReportHistoryItem(BaseModel):
    report_id: str
    start_date: str
    end_date: str
    format: str
    total_invoices: int
    total_amount: float
    total_gst: float
    created_at: str
    files: ReportFileLinks


class ReportHistoryResponse(BaseModel):
    items: list[ReportHistoryItem]
    total: int
    page: int
    limit: int
    total_pages: int


class SendToCAResponse(BaseModel):
    success: bool
    message: str
    report_id: Optional[str] = None
    ca_email: Optional[str] = None


class AutoSendSettingsResponse(BaseModel):
    enabled: bool
    ca_name: Optional[str] = None
    ca_email: Optional[str] = None
    schedule: str = "monthly"
