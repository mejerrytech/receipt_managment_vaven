from uuid import UUID

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.reports.dependencies import require_user_id
from api.reports.schemas import (
    AutoSendSettingsRequest,
    AutoSendSettingsResponse,
    GenerateReportRequest,
    GenerateReportResponse,
    ReportFileLinks,
    ReportHistoryItem,
    ReportHistoryResponse,
    ReportPreviewResponse,
    SendToCARequest,
    SendToCAResponse,
)
from api.reports.utils import (
    add_history_record,
    build_preview,
    category_breakdown,
    generate_report_files,
    get_auto_send_settings,
    get_history_record,
    get_latest_report,
    itc_summary,
    list_history,
    new_report_id,
    report_file_path,
    save_auto_send_settings,
    send_report_email,
)

logger = logging.getLogger("reports_api")

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _file_links(user_id: UUID, report_id: str, fmt: str) -> ReportFileLinks:
    pdf = f"/api/reports/download/{report_id}?user_id={user_id}&file_format=pdf"
    csv = f"/api/reports/download/{report_id}?user_id={user_id}&file_format=csv"
    if fmt == "pdf":
        return ReportFileLinks(pdf=pdf)
    if fmt == "csv":
        return ReportFileLinks(csv=csv)
    return ReportFileLinks(pdf=pdf, csv=csv)


@router.get("/preview", response_model=ReportPreviewResponse)
def report_preview(
    user_id: UUID = Depends(require_user_id),
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
) -> ReportPreviewResponse:
    """Report preview stats for the selected date range."""
    preview = build_preview(user_id, start_date, end_date)
    return ReportPreviewResponse(
        period_label=preview["period_label"],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        total_invoices=preview["total_invoices"],
        total_amount=preview["total_amount"],
        total_gst=preview["total_gst"],
        itc_eligible=preview["itc_eligible"],
        currency=preview["currency"],
    )


@router.post("/generate", response_model=GenerateReportResponse)
def generate_report(
    body: GenerateReportRequest,
    user_id: UUID = Depends(require_user_id),
) -> GenerateReportResponse:
    """Generate PDF, CSV, or both for the selected period."""
    if body.start_date > body.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    preview = build_preview(user_id, body.start_date, body.end_date)
    rows = preview.pop("rows", [])

    if body.all_confirmed_receipts is False:
        rows = []

    itc = itc_summary(rows) if body.itc_summary else None
    categories = category_breakdown(rows) if body.category_breakdown else None

    report_id = new_report_id()
    generate_report_files(
        user_id,
        report_id,
        preview,
        rows,
        body.format,
        include_itc=body.itc_summary,
        include_categories=body.category_breakdown,
        itc=itc,
        categories=categories,
    )

    files = _file_links(user_id, report_id, body.format)
    record = {
        "report_id": report_id,
        "start_date": body.start_date,
        "end_date": body.end_date,
        "format": body.format,
        "total_invoices": preview["total_invoices"],
        "total_amount": preview["total_amount"],
        "total_gst": preview["total_gst"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files.model_dump(),
    }
    add_history_record(user_id, record)

    return GenerateReportResponse(
        success=True,
        report_id=report_id,
        format=body.format,
        files=files,
        preview=ReportPreviewResponse(
            period_label=preview["period_label"],
            start_date=preview["start_date"],
            end_date=preview["end_date"],
            total_invoices=preview["total_invoices"],
            total_amount=preview["total_amount"],
            total_gst=preview["total_gst"],
            itc_eligible=preview["itc_eligible"],
            currency=preview["currency"],
        ),
        message="Report generated successfully",
    )


@router.get("/history", response_model=ReportHistoryResponse)
def report_history(
    user_id: UUID = Depends(require_user_id),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
) -> ReportHistoryResponse:
    """Previously generated reports."""
    data = list_history(user_id, page=page, limit=limit)
    items = [
        ReportHistoryItem(
            report_id=i["report_id"],
            start_date=i["start_date"],
            end_date=i["end_date"],
            format=i["format"],
            total_invoices=i.get("total_invoices", 0),
            total_amount=i.get("total_amount", 0),
            total_gst=i.get("total_gst", 0),
            created_at=i["created_at"],
            files=ReportFileLinks(**i.get("files", {})),
        )
        for i in data["items"]
    ]
    return ReportHistoryResponse(
        items=items,
        total=data["total"],
        page=data["page"],
        limit=data["limit"],
        total_pages=data["total_pages"],
    )


@router.get("/download/{report_id}")
def download_report(
    report_id: str,
    user_id: UUID = Depends(require_user_id),
    file_format: str = Query("pdf", alias="file_format", description="pdf or csv"),
):
    """Download a generated report file."""
    record = get_history_record(user_id, report_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")

    ext = "csv" if file_format.lower() == "csv" else "pdf"
    path = report_file_path(user_id, report_id, ext)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Report {ext.upper()} file not found")

    media = "text/csv" if ext == "csv" else "application/pdf"
    return FileResponse(path, media_type=media, filename=f"report_{report_id[:8]}.{ext}")


@router.post("/send-to-ca", response_model=SendToCAResponse)
def send_report_to_ca(
    body: SendToCARequest,
    user_id: UUID = Depends(require_user_id),
) -> SendToCAResponse:
    """Email the latest or a specific report to the CA."""
    record = get_history_record(user_id, body.report_id) if body.report_id else get_latest_report(user_id)
    if not record:
        return SendToCAResponse(success=False, message="No report found to send")

    report_id = record["report_id"]
    attachments = []
    for ext in ("pdf", "csv"):
        path = report_file_path(user_id, report_id, ext)
        if path.is_file():
            attachments.append(path)

    if not attachments:
        return SendToCAResponse(success=False, message="Report files missing on disk")

    period = f"{record['start_date']} to {record['end_date']}"
    body_text = (
        f"Dear {body.ca_name},\n\n"
        f"Please find the expense report for period {period} attached.\n\n"
        f"Total invoices: {record.get('total_invoices', 0)}\n"
        f"Total amount: {record.get('total_amount', 0)}\n"
        f"Total GST: {record.get('total_gst', 0)}\n"
    )
    ok, msg = send_report_email(
        ca_name=body.ca_name,
        ca_email=str(body.ca_email),
        subject=f"Expense Report — {period}",
        body=body_text,
        attachments=attachments,
    )
    return SendToCAResponse(
        success=ok,
        message=msg,
        report_id=report_id,
        ca_email=str(body.ca_email),
    )


@router.get("/auto-send", response_model=AutoSendSettingsResponse)
def get_auto_send(user_id: UUID = Depends(require_user_id)) -> AutoSendSettingsResponse:
    """Auto Send to CA settings."""
    settings = get_auto_send_settings(user_id)
    return AutoSendSettingsResponse(
        enabled=bool(settings.get("enabled")),
        ca_name=settings.get("ca_name"),
        ca_email=settings.get("ca_email"),
        schedule=settings.get("schedule", "monthly"),
    )


@router.put("/auto-send", response_model=AutoSendSettingsResponse)
def update_auto_send(
    body: AutoSendSettingsRequest,
    user_id: UUID = Depends(require_user_id),
) -> AutoSendSettingsResponse:
    """Save Auto Send to CA settings."""
    settings = save_auto_send_settings(
        user_id,
        {
            "enabled": body.enabled,
            "ca_name": body.ca_name,
            "ca_email": str(body.ca_email) if body.ca_email else None,
            "schedule": body.schedule,
        },
    )
    return AutoSendSettingsResponse(
        enabled=bool(settings.get("enabled")),
        ca_name=settings.get("ca_name"),
        ca_email=settings.get("ca_email"),
        schedule=settings.get("schedule", "monthly"),
    )
