from typing import Literal, Optional

from pydantic import BaseModel, Field


ReportFormat = Literal["pdf", "csv", "both"]


class GenerateReportRequest(BaseModel):
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    format: ReportFormat = "pdf"
    all_confirmed_receipts: bool = True
    itc_summary: bool = False
    category_breakdown: bool = False


class SendToCARequest(BaseModel):
    ca_name: str
    ca_email: str
    report_id: Optional[str] = Field(None, description="Specific report id; latest if omitted")


class AutoSendSettingsRequest(BaseModel):
    enabled: bool = False
    ca_name: Optional[str] = None
    ca_email: Optional[str] = None
    schedule: Literal["monthly", "weekly"] = "monthly"
