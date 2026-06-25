"""
Expense report generation service.

Aggregates expense data from Documents + UserTextEntries and formats
WhatsApp-friendly daily and weekly report messages.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from shared.database import DatabaseService, Document, UserTextEntry, User, get_db

logger = logging.getLogger("report_service")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _week_range(ref: date) -> Tuple[date, date]:
    """Return (monday, sunday) of the ISO week containing *ref*."""
    monday = ref - timedelta(days=ref.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _format_currency(amount: float, currency: str) -> str:
    symbol_map = {
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "AUD": "A$",
        "CAD": "C$",
    }
    symbol = symbol_map.get((currency or "INR").upper(), (currency or "INR") + " ")
    return f"{symbol}{amount:,.0f}"


def _display_name(user: User) -> str:
    parts = [p for p in [user.first_name, user.last_name] if p and p.strip()]
    name = " ".join(parts).strip()
    # WhatsApp users are created with first_name="WhatsApp"; fall back to username
    if not name or name == "WhatsApp":
        if user.username and not user.username.startswith("wa_"):
            return user.username
        return "there"
    return name


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_expenses_for_date_range(
    user_id: int,
    start: date,
    end: date,
) -> List[Dict[str, Any]]:
    """
    Return expense rows from both Documents and UserTextEntries
    where the expense date falls within [start, end] inclusive.
    Only rows with a non-None amount > 0 are included.
    """
    db = get_db()
    try:
        rows: List[Dict[str, Any]] = []

        # --- Documents (OCR receipts) ---
        docs = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .all()
        )
        for doc in docs:
            if not doc.total_amount or doc.total_amount <= 0:
                continue
            # Resolve expense date: prefer document_date field, fall back to created_at
            expense_date = _parse_date(doc.document_date) or (
                doc.created_at.date() if doc.created_at else None
            )
            if expense_date is None or not (start <= expense_date <= end):
                continue
            cat = DatabaseService.normalize_expense_category_label(doc.expense_category)
            rows.append({
                "category": cat,
                "amount": doc.total_amount,
                "currency": (doc.currency or "INR").upper(),
                "date": expense_date,
            })

        # --- UserTextEntries (manual text expenses) ---
        texts = (
            db.query(UserTextEntry)
            .filter(UserTextEntry.user_id == user_id)
            .all()
        )
        for entry in texts:
            if not entry.amount or entry.amount <= 0:
                continue
            expense_date = entry.created_at.date() if entry.created_at else None
            if expense_date is None or not (start <= expense_date <= end):
                continue
            cat = DatabaseService.normalize_expense_category_label(entry.expense_category)
            rows.append({
                "category": cat,
                "amount": entry.amount,
                "currency": (entry.currency or "INR").upper(),
                "date": expense_date,
            })

        return rows
    finally:
        db.close()


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse common date string formats into a date object."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    # Try ISO datetime prefix (e.g. "2025-06-25T...")
    try:
        return date.fromisoformat(value[:10])
    except Exception:
        return None


def _aggregate_by_category(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Returns {category: {currency: total_amount}}.
    When multiple currencies exist we keep them separate.
    """
    agg: Dict[str, Dict[str, float]] = {}
    for row in rows:
        cat = row["category"]
        cur = row["currency"]
        agg.setdefault(cat, {})
        agg[cat][cur] = agg[cat].get(cur, 0.0) + row["amount"]
    return agg


def _dominant_currency(rows: List[Dict[str, Any]]) -> str:
    """Pick the most-used currency by transaction count."""
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["currency"]] = counts.get(row["currency"], 0) + 1
    return max(counts, key=counts.__getitem__) if counts else "INR"


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def _category_lines(agg: Dict[str, Dict[str, float]]) -> List[str]:
    lines = []
    for cat, currencies in sorted(agg.items()):
        for currency, total in sorted(currencies.items()):
            lines.append(f"• {cat}: {_format_currency(total, currency)}")
    return lines


def build_daily_report(user: User, report_date: Optional[date] = None) -> Optional[str]:
    """
    Build the daily WhatsApp report message for a user.
    Returns None if there are no expenses to report.
    """
    target = report_date or _today_utc()
    rows = get_expenses_for_date_range(user.id, target, target)
    if not rows:
        return None

    agg = _aggregate_by_category(rows)
    dominant_cur = _dominant_currency(rows)
    grand_total = sum(r["amount"] for r in rows if r["currency"] == dominant_cur)
    # If multi-currency just sum all (best-effort)
    if len(set(r["currency"] for r in rows)) > 1:
        grand_total = sum(r["amount"] for r in rows)

    name = _display_name(user)
    date_str = target.strftime("%d %B %Y")
    category_block = "\n".join(_category_lines(agg))

    lines = [
        "📊 *Daily Expense Report*",
        "",
        f"Hi {name},",
        "Here is your expense summary for today:",
        "",
        category_block,
        "",
        f"💰 *Total Expenses Today: {_format_currency(grand_total, dominant_cur)}*",
        f"📅 Date: {date_str}",
        "",
        "Thank you for using Expense Planner.",
    ]
    return "\n".join(lines)


def build_weekly_report(user: User, ref_date: Optional[date] = None) -> Optional[str]:
    """
    Build the weekly WhatsApp report message for a user.
    Returns None if there are no expenses to report.
    """
    start, end = _week_range(ref_date or _today_utc())
    rows = get_expenses_for_date_range(user.id, start, end)
    if not rows:
        return None

    agg = _aggregate_by_category(rows)
    dominant_cur = _dominant_currency(rows)
    grand_total = sum(r["amount"] for r in rows if r["currency"] == dominant_cur)
    if len(set(r["currency"] for r in rows)) > 1:
        grand_total = sum(r["amount"] for r in rows)

    # Highest spending category (in dominant currency)
    cat_totals = {
        cat: sum(amounts.get(dominant_cur, 0) for amounts in [currencies])
        for cat, currencies in agg.items()
    }
    top_cat = max(cat_totals, key=cat_totals.__getitem__) if cat_totals else None
    top_amount = cat_totals[top_cat] if top_cat else 0.0

    name = _display_name(user)
    period = f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"
    category_block = "\n".join(_category_lines(agg))

    lines = [
        "📈 *Weekly Expense Report*",
        "",
        f"Hi {name},",
        "Here is your expense summary for this week:",
        "",
        category_block,
        "",
        f"💰 *Total Expenses This Week: {_format_currency(grand_total, dominant_cur)}*",
    ]
    if top_cat:
        lines.append(
            f"🏆 Highest Spending Category: {top_cat} ({_format_currency(top_amount, dominant_cur)})"
        )
    lines += [
        f"📅 Period: {period}",
        "",
        "Thank you for using Expense Planner.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User resolver  (WhatsApp users only)
# ---------------------------------------------------------------------------

def get_all_whatsapp_users() -> List[Dict[str, Any]]:
    """
    Return all users created via the WhatsApp bot.

    WhatsApp users are identified by username starting with 'wa_'
    (set in bot_whatsapp/bot.py: username=f"wa_{phone_digits[-10:]}").
    telegram_id for these users equals int(phone_digits).
    """
    db = get_db()
    try:
        users = (
            db.query(User)
            .filter(User.username.like("wa_%"))
            .all()
        )
        result = []
        for u in users:
            # Reconstruct phone digits from telegram_id
            phone_digits = str(u.telegram_id)
            result.append({"user": u, "phone": phone_digits})
        return result
    finally:
        db.close()
