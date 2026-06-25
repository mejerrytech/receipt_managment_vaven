import os

from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "4"))
CELERY_TASK_MAX_RETRIES = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
CELERY_TASK_RETRY_BACKOFF_SECONDS = int(os.getenv("CELERY_TASK_RETRY_BACKOFF_SECONDS", "10"))
OCR_TASK_SOFT_TIME_LIMIT = int(os.getenv("OCR_TASK_SOFT_TIME_LIMIT", "120"))
OCR_TASK_TIME_LIMIT = int(os.getenv("OCR_TASK_TIME_LIMIT", "180"))

# Report schedule (24-hour UTC). Default: 22:00 UTC = 10 PM IST-offset-adjusted.
# Override via env if your users are in a different timezone.
DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR_UTC", "16"))    # 16:30 UTC = 22:00 IST
DAILY_REPORT_MINUTE = int(os.getenv("DAILY_REPORT_MINUTE_UTC", "30"))
WEEKLY_REPORT_DAY = os.getenv("WEEKLY_REPORT_DAY_OF_WEEK", "0")       # 0 = Monday
WEEKLY_REPORT_HOUR = int(os.getenv("WEEKLY_REPORT_HOUR_UTC", "16"))
WEEKLY_REPORT_MINUTE = int(os.getenv("WEEKLY_REPORT_MINUTE_UTC", "30"))

celery_app = Celery(
    "local_expensebot",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=CELERY_WORKER_CONCURRENCY,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_queue="ocr_jobs",
    task_routes={
        "shared.tasks.ocr_tasks.process_pending_ocr": {"queue": "ocr_jobs"},
        "shared.tasks.report_tasks.send_daily_whatsapp_reports": {"queue": "reports"},
        "shared.tasks.report_tasks.send_weekly_whatsapp_reports": {"queue": "reports"},
    },
    task_default_retry_delay=CELERY_TASK_RETRY_BACKOFF_SECONDS,
    # Celery Beat periodic schedule
    beat_schedule={
        "daily-whatsapp-expense-report": {
            "task": "shared.tasks.report_tasks.send_daily_whatsapp_reports",
            "schedule": crontab(hour=DAILY_REPORT_HOUR, minute=DAILY_REPORT_MINUTE),
            "options": {"queue": "reports"},
        },
        "weekly-whatsapp-expense-report": {
            "task": "shared.tasks.report_tasks.send_weekly_whatsapp_reports",
            "schedule": crontab(
                hour=WEEKLY_REPORT_HOUR,
                minute=WEEKLY_REPORT_MINUTE,
                day_of_week=WEEKLY_REPORT_DAY,
            ),
            "options": {"queue": "reports"},
        },
    },
)

celery_app.autodiscover_tasks(["shared.tasks"])
