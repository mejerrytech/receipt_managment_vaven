#!/usr/bin/env bash
set -euo pipefail

if [ -f "env/bin/activate" ]; then
  # shellcheck source=/dev/null
  source env/bin/activate
fi

exec celery -A shared.celery_app:celery_app worker -Q ocr_jobs --loglevel=info --concurrency="${CELERY_WORKER_CONCURRENCY:-4}"
