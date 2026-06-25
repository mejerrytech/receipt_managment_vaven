"""Dashboard API routes."""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query

from api.services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/total-pending")
def total_pending(user_id: int = Query(...)):
    """Total pending receipts awaiting confirmation."""
    return DashboardService.get_total_pending(user_id)


@router.get("/month-spend")
def month_spend(
    user_id: int = Query(...),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    month: Optional[int] = Query(None, ge=1, le=12),
):
    """Total spend for the current or selected month (documents + text entries)."""
    return DashboardService.get_month_spend(user_id, year=year, month=month)


@router.get("/gst-detected")
def gst_detected(
    user_id: int = Query(...),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    month: Optional[int] = Query(None, ge=1, le=12),
):
    """GST totals detected from receipt OCR for the month."""
    return DashboardService.get_gst_detected(user_id, year=year, month=month)


@router.get("/spend-calendar")
def spend_calendar(
    user_id: int = Query(...),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    month: Optional[int] = Query(None, ge=1, le=12),
):
    """Daily spend totals for calendar heatmap (select a date for category drill-down)."""
    return DashboardService.get_spend_calendar(user_id, year=year, month=month)


@router.get("/spend-calendar/by-date")
def spend_calendar_by_date(
    user_id: int = Query(...),
    date_str: str = Query(..., alias="date", description="YYYY-MM-DD"),
):
    """Spend broken down by category for a single calendar date."""
    try:
        target = date.fromisoformat(date_str)
    except ValueError:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Invalid date; use YYYY-MM-DD")
    return DashboardService.get_spend_by_date_categories(user_id, target)


@router.get("/monthly-spend-trend")
def monthly_spend_trend(user_id: int = Query(...)):
    """Compare current month spend vs previous month."""
    return DashboardService.get_monthly_spend_trend(user_id)


@router.get("/spend-by-category")
def spend_by_category(
    user_id: int = Query(...),
    period: str = Query("month", pattern="^(month|week)$"),
):
    """Total spend grouped by category — `month` (calendar month) or `week` (last 7 days)."""
    return DashboardService.get_spend_by_category(user_id, period=period)


@router.get("/recent-activity")
def recent_activity(
    user_id: int = Query(...),
    limit: int = Query(30, ge=1, le=200),
):
    """Recent receipts and manual text expenses."""
    return DashboardService.get_recent_activity(user_id, limit=limit)
