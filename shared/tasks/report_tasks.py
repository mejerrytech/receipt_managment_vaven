"""
Celery tasks for automated WhatsApp expense reporting.

Schedules:
  - send_daily_whatsapp_reports   → every day at 22:00 UTC
  - send_weekly_whatsapp_reports  → every Monday at 22:00 UTC (covers prior week)

Each task iterates over all WhatsApp users and sends a personalised report
via Twilio if the user has any expenses in the reporting period.
"""

import logging
import os
from datetime import date

from shared.celery_app import celery_app
from shared.report_service import (
    build_daily_report,
    build_weekly_report,
    get_all_whatsapp_users,
)

logger = logging.getLogger("report_tasks")


def _send_report(phone: str, message: str) -> bool:
    """Send a WhatsApp message; returns True on success."""
    try:
        from bot_whatsapp.sender import send_whatsapp_message
        send_whatsapp_message(to_number=phone, body=message)
        return True
    except Exception as exc:
        logger.error("Failed to send report to +%s: %s", phone, exc)
        return False


# ---------------------------------------------------------------------------
# Daily report task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="shared.tasks.report_tasks.send_daily_whatsapp_reports",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="reports",
)
def send_daily_whatsapp_reports(self, report_date_iso: str | None = None):
    """
    Send daily expense summaries to all WhatsApp users.
    report_date_iso: optional override in 'YYYY-MM-DD' format (for testing).
    """
    target = (
        date.fromisoformat(report_date_iso) if report_date_iso else None
    )

    wa_users = get_all_whatsapp_users()
    if not wa_users:
        logger.info("daily_report: no WhatsApp users found")
        return {"sent": 0, "skipped": 0}

    sent = 0
    skipped = 0
    for entry in wa_users:
        user = entry["user"]
        phone = entry["phone"]
        try:
            message = build_daily_report(user, report_date=target)
            if not message:
                logger.debug("daily_report: no expenses for user_id=%s", user.id)
                skipped += 1
                continue
            if _send_report(phone, message):
                sent += 1
                logger.info("daily_report sent to user_id=%s phone=+%s", user.id, phone)
            else:
                skipped += 1
        except Exception as exc:
            logger.exception("daily_report: error for user_id=%s: %s", user.id, exc)
            skipped += 1

    logger.info("daily_report complete: sent=%s skipped=%s", sent, skipped)
    return {"sent": sent, "skipped": skipped}


# ---------------------------------------------------------------------------
# Weekly report task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="shared.tasks.report_tasks.send_weekly_whatsapp_reports",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="reports",
)
def send_weekly_whatsapp_reports(self, ref_date_iso: str | None = None):
    """
    Send weekly expense summaries to all WhatsApp users.
    ref_date_iso: optional override in 'YYYY-MM-DD' format (for testing).
    """
    ref = (
        date.fromisoformat(ref_date_iso) if ref_date_iso else None
    )

    wa_users = get_all_whatsapp_users()
    if not wa_users:
        logger.info("weekly_report: no WhatsApp users found")
        return {"sent": 0, "skipped": 0}

    sent = 0
    skipped = 0
    for entry in wa_users:
        user = entry["user"]
        phone = entry["phone"]
        try:
            message = build_weekly_report(user, ref_date=ref)
            if not message:
                logger.debug("weekly_report: no expenses for user_id=%s", user.id)
                skipped += 1
                continue
            if _send_report(phone, message):
                sent += 1
                logger.info("weekly_report sent to user_id=%s phone=+%s", user.id, phone)
            else:
                skipped += 1
        except Exception as exc:
            logger.exception("weekly_report: error for user_id=%s: %s", user.id, exc)
            skipped += 1

    logger.info("weekly_report complete: sent=%s skipped=%s", sent, skipped)
    return {"sent": sent, "skipped": skipped}
