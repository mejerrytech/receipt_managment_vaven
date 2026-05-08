import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "4"))
CELERY_TASK_MAX_RETRIES = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
CELERY_TASK_RETRY_BACKOFF_SECONDS = int(os.getenv("CELERY_TASK_RETRY_BACKOFF_SECONDS", "10"))
OCR_TASK_SOFT_TIME_LIMIT = int(os.getenv("OCR_TASK_SOFT_TIME_LIMIT", "120"))
OCR_TASK_TIME_LIMIT = int(os.getenv("OCR_TASK_TIME_LIMIT", "180"))

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
    },
    task_default_retry_delay=CELERY_TASK_RETRY_BACKOFF_SECONDS,
)

celery_app.autodiscover_tasks(["shared.tasks"])
