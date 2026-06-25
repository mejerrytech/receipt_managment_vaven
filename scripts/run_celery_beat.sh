#!/usr/bin/env bash
# Celery Beat scheduler — triggers periodic report tasks.
# Run this alongside the Celery worker (run_celery_worker.sh).
set -euo pipefail

if [ -f "env/bin/activate" ]; then
  source env/bin/activate
fi

exec celery -A shared.celery_app:celery_app beat --loglevel=info
