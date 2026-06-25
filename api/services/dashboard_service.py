"""Dashboard aggregations for receipts and text expenses."""

from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from shared.database import DatabaseService, engine
from shared.receipt_card_formatter import compact_ocr_payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start, end


def _week_bounds(reference: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    ref = reference or _utc_now()
    start_of_day = ref.replace(hour=0, minute=0, second=0, microsecond=0)
    # Rolling last 7 days including today
    start = start_of_day - timedelta(days=6)
    end = start_of_day + timedelta(days=1)
    return start, end


def _parse_document_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _effective_expense_date(created_at: Optional[datetime], document_date: Optional[str]) -> date:
    parsed = _parse_document_date(document_date)
    if parsed:
        return parsed
    if created_at:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at.date()
    return _utc_now().date()


def _float_or_zero(value: Any) -> float:
    if value in (None, "", [], {}):
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _gst_from_extracted(extracted_data: Optional[str]) -> Dict[str, float]:
    if not extracted_data:
        return {"cgst": 0.0, "sgst": 0.0, "igst": 0.0, "total_gst": 0.0}
    try:
        payload = json.loads(extracted_data) if isinstance(extracted_data, str) else extracted_data
    except (json.JSONDecodeError, TypeError):
        return {"cgst": 0.0, "sgst": 0.0, "igst": 0.0, "total_gst": 0.0}
    fields = compact_ocr_payload(payload if isinstance(payload, dict) else {})
    cgst = _float_or_zero(fields.get("cgst"))
    sgst = _float_or_zero(fields.get("sgst"))
    igst = _float_or_zero(fields.get("igst"))
    tax_amount = _float_or_zero(fields.get("tax_amount"))
    total_gst = cgst + sgst + igst
    if total_gst == 0.0 and tax_amount > 0:
        total_gst = tax_amount
    return {"cgst": cgst, "sgst": sgst, "igst": igst, "total_gst": total_gst}


def _pending_amount_from_json(extracted_data: Optional[str]) -> float:
    if not extracted_data:
        return 0.0
    try:
        data = json.loads(extracted_data)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    amounts = data.get("amounts") if isinstance(data.get("amounts"), dict) else {}
    total = amounts.get("total")
    if total is None:
        total = data.get("total_amount")
    return _float_or_zero(total)


class DashboardService:
    """Compute dashboard metrics for a single user."""

    @staticmethod
    def get_total_pending(user_id: int) -> Dict[str, Any]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, status, extracted_data, file_name, created_at
                    FROM pending_documents
                    WHERE user_id = :uid
                      AND status IN ('pending', 'processing', 'ready')
                    ORDER BY created_at DESC
                    """
                ),
                {"uid": user_id},
            ).mappings().all()

        total_amount = 0.0
        items: List[Dict[str, Any]] = []
        for row in rows:
            amount = _pending_amount_from_json(row["extracted_data"])
            total_amount += amount
            created = row["created_at"]
            items.append(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "file_name": row["file_name"],
                    "amount": round(amount, 2),
                    "created_at": created.isoformat() if created else None,
                }
            )

        return {
            "count": len(items),
            "total_amount": round(total_amount, 2),
            "currency": "INR",
            "items": items,
        }

    @staticmethod
    def get_month_spend(user_id: int, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        now = _utc_now()
        year = year or now.year
        month = month or now.month
        start, end = _month_bounds(year, month)

        doc_total, doc_count = DashboardService._sum_documents_in_range(user_id, start, end)
        text_total, text_count = DashboardService._sum_text_entries_in_range(user_id, start, end)

        return {
            "year": year,
            "month": month,
            "label": datetime(year, month, 1).strftime("%B %Y"),
            "total_amount": round(doc_total + text_total, 2),
            "document_amount": round(doc_total, 2),
            "text_entry_amount": round(text_total, 2),
            "transaction_count": doc_count + text_count,
            "currency": "INR",
        }

    @staticmethod
    def get_gst_detected(user_id: int, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        now = _utc_now()
        year = year or now.year
        month = month or now.month
        start, end = _month_bounds(year, month)

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, title, file_name, gstin, extracted_data, created_at
                    FROM documents
                    WHERE user_id = :uid
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().all()

        cgst_total = sgst_total = igst_total = 0.0
        docs_with_gst = 0
        receipts: List[Dict[str, Any]] = []

        for row in rows:
            gst = _gst_from_extracted(row["extracted_data"])
            has_gst = bool(row["gstin"]) or gst["total_gst"] > 0
            if not has_gst:
                continue
            docs_with_gst += 1
            cgst_total += gst["cgst"]
            sgst_total += gst["sgst"]
            igst_total += gst["igst"]
            created = row["created_at"]
            receipts.append(
                {
                    "id": row["id"],
                    "title": row["title"] or row["file_name"] or "Receipt",
                    "gstin": row["gstin"],
                    "cgst": round(gst["cgst"], 2),
                    "sgst": round(gst["sgst"], 2),
                    "igst": round(gst["igst"], 2),
                    "total_gst": round(gst["total_gst"], 2),
                    "created_at": created.isoformat() if created else None,
                }
            )

        return {
            "year": year,
            "month": month,
            "label": datetime(year, month, 1).strftime("%B %Y"),
            "receipts_with_gst": docs_with_gst,
            "cgst_total": round(cgst_total, 2),
            "sgst_total": round(sgst_total, 2),
            "igst_total": round(igst_total, 2),
            "total_gst": round(cgst_total + sgst_total + igst_total, 2),
            "currency": "INR",
            "receipts": receipts,
        }

    @staticmethod
    def get_spend_calendar(user_id: int, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        now = _utc_now()
        year = year or now.year
        month = month or now.month
        start, end = _month_bounds(year, month)
        days_in_month = monthrange(year, month)[1]

        daily: Dict[str, Dict[str, Any]] = {
            f"{year:04d}-{month:02d}-{day:02d}": {"date": f"{year:04d}-{month:02d}-{day:02d}", "total": 0.0, "count": 0}
            for day in range(1, days_in_month + 1)
        }

        with engine.connect() as conn:
            docs = conn.execute(
                text(
                    """
                    SELECT total_amount, document_date, created_at
                    FROM documents
                    WHERE user_id = :uid
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().all()
            texts = conn.execute(
                text(
                    """
                    SELECT amount, created_at
                    FROM user_text_entries
                    WHERE user_id = :uid
                      AND amount IS NOT NULL
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().all()

        for doc in docs:
            day_key = _effective_expense_date(doc["created_at"], doc["document_date"]).isoformat()
            if day_key not in daily:
                continue
            amount = _float_or_zero(doc["total_amount"])
            daily[day_key]["total"] += amount
            daily[day_key]["count"] += 1

        for entry in texts:
            created = entry["created_at"]
            if not created:
                continue
            day_key = _effective_expense_date(created, None).isoformat()
            if day_key not in daily:
                continue
            amount = _float_or_zero(entry["amount"])
            daily[day_key]["total"] += amount
            daily[day_key]["count"] += 1

        days = []
        for day in range(1, days_in_month + 1):
            key = f"{year:04d}-{month:02d}-{day:02d}"
            row = daily[key]
            days.append(
                {
                    "date": row["date"],
                    "total": round(row["total"], 2),
                    "count": row["count"],
                    "has_spend": row["count"] > 0,
                }
            )

        month_total = round(sum(d["total"] for d in days), 2)
        return {
            "year": year,
            "month": month,
            "label": datetime(year, month, 1).strftime("%B %Y"),
            "month_total": month_total,
            "currency": "INR",
            "days": days,
        }

    @staticmethod
    def get_spend_by_date_categories(user_id: int, target_date: date) -> Dict[str, Any]:
        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        categories: Dict[str, float] = {}
        items: List[Dict[str, Any]] = []

        with engine.connect() as conn:
            docs = conn.execute(
                text(
                    """
                    SELECT id, title, file_name, total_amount, expense_category,
                           document_date, created_at, source
                    FROM documents
                    WHERE user_id = :uid
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start - timedelta(days=31), "end": end + timedelta(days=31)},
            ).mappings().all()
            texts = conn.execute(
                text(
                    """
                    SELECT id, text, amount, expense_category, created_at, source
                    FROM user_text_entries
                    WHERE user_id = :uid
                      AND amount IS NOT NULL
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().all()

        for doc in docs:
            if _effective_expense_date(doc["created_at"], doc["document_date"]) != target_date:
                continue
            cat = DatabaseService.normalize_expense_category_label(doc["expense_category"])
            amount = _float_or_zero(doc["total_amount"])
            categories[cat] = categories.get(cat, 0.0) + amount
            created = doc["created_at"]
            items.append(
                {
                    "type": "receipt",
                    "id": doc["id"],
                    "title": doc["title"] or doc["file_name"] or "Receipt",
                    "amount": round(amount, 2),
                    "expense_category": cat,
                    "channel": doc["source"] or "telegram",
                    "created_at": created.isoformat() if created else None,
                }
            )

        for entry in texts:
            created = entry["created_at"]
            if not created or _effective_expense_date(created, None) != target_date:
                continue
            cat = DatabaseService.normalize_expense_category_label(entry["expense_category"])
            amount = _float_or_zero(entry["amount"])
            categories[cat] = categories.get(cat, 0.0) + amount
            items.append(
                {
                    "type": "text",
                    "id": entry["id"],
                    "title": (entry["text"] or "")[:120],
                    "amount": round(amount, 2),
                    "expense_category": cat,
                    "channel": entry["source"] or "telegram",
                    "created_at": created.isoformat() if created else None,
                }
            )

        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        breakdown = [
            {"expense_category": cat, "total": round(total, 2)}
            for cat, total in sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
        ]

        return {
            "date": target_date.isoformat(),
            "total_amount": round(sum(categories.values()), 2),
            "currency": "INR",
            "categories": breakdown,
            "items": items,
        }

    @staticmethod
    def get_monthly_spend_trend(user_id: int) -> Dict[str, Any]:
        now = _utc_now()
        cur_year, cur_month = now.year, now.month
        if cur_month == 1:
            prev_year, prev_month = cur_year - 1, 12
        else:
            prev_year, prev_month = cur_year, cur_month - 1

        current = DashboardService.get_month_spend(user_id, cur_year, cur_month)
        previous = DashboardService.get_month_spend(user_id, prev_year, prev_month)

        cur_amt = current["total_amount"]
        prev_amt = previous["total_amount"]
        if prev_amt > 0:
            change_pct = round(((cur_amt - prev_amt) / prev_amt) * 100, 1)
        elif cur_amt > 0:
            change_pct = 100.0
        else:
            change_pct = 0.0

        return {
            "current_month": {
                "year": cur_year,
                "month": cur_month,
                "label": current["label"],
                "total_amount": cur_amt,
                "transaction_count": current["transaction_count"],
            },
            "previous_month": {
                "year": prev_year,
                "month": prev_month,
                "label": previous["label"],
                "total_amount": prev_amt,
                "transaction_count": previous["transaction_count"],
            },
            "difference": round(cur_amt - prev_amt, 2),
            "change_percent": change_pct,
            "currency": "INR",
        }

    @staticmethod
    def get_spend_by_category(user_id: int, period: str = "month") -> Dict[str, Any]:
        period_norm = (period or "month").strip().lower()
        now = _utc_now()
        if period_norm == "week":
            start, end = _week_bounds(now)
            label = "Last 7 days"
        else:
            start, end = _month_bounds(now.year, now.month)
            label = now.strftime("%B %Y")
            period_norm = "month"

        categories: Dict[str, float] = {}
        count = 0

        with engine.connect() as conn:
            docs = conn.execute(
                text(
                    """
                    SELECT expense_category, total_amount
                    FROM documents
                    WHERE user_id = :uid
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().all()
            texts = conn.execute(
                text(
                    """
                    SELECT expense_category, amount
                    FROM user_text_entries
                    WHERE user_id = :uid
                      AND amount IS NOT NULL
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().all()

        for doc in docs:
            cat = DatabaseService.normalize_expense_category_label(doc["expense_category"])
            amount = _float_or_zero(doc["total_amount"])
            categories[cat] = categories.get(cat, 0.0) + amount
            count += 1

        for entry in texts:
            cat = DatabaseService.normalize_expense_category_label(entry["expense_category"])
            amount = _float_or_zero(entry["amount"])
            categories[cat] = categories.get(cat, 0.0) + amount
            count += 1

        breakdown = [
            {
                "expense_category": cat,
                "total": round(total, 2),
                "percent": 0.0,
            }
            for cat, total in sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
        ]
        grand_total = sum(c["total"] for c in breakdown)
        if grand_total > 0:
            for row in breakdown:
                row["percent"] = round((row["total"] / grand_total) * 100, 1)

        return {
            "period": period_norm,
            "label": label,
            "total_amount": round(grand_total, 2),
            "transaction_count": count,
            "currency": "INR",
            "categories": breakdown,
        }

    @staticmethod
    def get_recent_activity(user_id: int, limit: int = 30) -> Dict[str, Any]:
        items = DatabaseService.get_user_expense_items(user_id=user_id, limit=limit)
        activity = []
        for item in items:
            activity.append(
                {
                    "type": "receipt" if item.get("source") == "document" else "text",
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "amount": item.get("payment"),
                    "currency": item.get("currency") or "INR",
                    "expense_category": item.get("expense_category"),
                    "vendor": item.get("vendor"),
                    "channel": item.get("channel"),
                    "date": item.get("date"),
                    "created_at": item.get("created_at"),
                }
            )
        return {"count": len(activity), "items": activity}

    @staticmethod
    def _sum_documents_in_range(user_id: int, start: datetime, end: datetime) -> Tuple[float, int]:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT COALESCE(SUM(total_amount), 0) AS total, COUNT(*) AS cnt
                    FROM documents
                    WHERE user_id = :uid
                      AND total_amount IS NOT NULL
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().one()
        return _float_or_zero(row["total"]), int(row["cnt"] or 0)

    @staticmethod
    def _sum_text_entries_in_range(user_id: int, start: datetime, end: datetime) -> Tuple[float, int]:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT COALESCE(SUM(amount), 0) AS total, COUNT(*) AS cnt
                    FROM user_text_entries
                    WHERE user_id = :uid
                      AND amount IS NOT NULL
                      AND created_at >= :start AND created_at < :end
                    """
                ),
                {"uid": user_id, "start": start, "end": end},
            ).mappings().one()
        return _float_or_zero(row["total"]), int(row["cnt"] or 0)
