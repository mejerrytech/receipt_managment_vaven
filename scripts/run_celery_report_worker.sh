#!/usr/bin/env bash
# Celery worker for the 'reports' queue (WhatsApp daily/weekly reports).
# Can run as a separate process or alongside the OCR worker.
set -euo pipefail

if [ -f "env/bin/activate" ]; then
  source env/bin/activate
fi

exec celery -A shared.celery_app:celery_app worker -Q reports --loglevel=info --concurrency=2
