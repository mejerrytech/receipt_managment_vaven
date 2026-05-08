import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import types

# Provide a lightweight telegram stub so tests don't require runtime dependency installed.
if "telegram" not in sys.modules:
    telegram_stub = types.ModuleType("telegram")
    telegram_stub.Bot = object
    telegram_stub.InlineKeyboardButton = object
    telegram_stub.InlineKeyboardMarkup = object
    sys.modules["telegram"] = telegram_stub

# Provide a lightweight celery stub so tests can import task module.
if "celery" not in sys.modules:
    celery_stub = types.ModuleType("celery")

    class _TaskWrapper:
        def __init__(self, fn):
            self.fn = fn
            self.run = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def delay(self, *args, **kwargs):
            return types.SimpleNamespace(id="test-job-id")

    class _FakeCelery:
        def __init__(self, *args, **kwargs):
            self.conf = {}

        def task(self, *args, **kwargs):
            def _decorator(fn):
                return _TaskWrapper(fn)
            return _decorator

        def autodiscover_tasks(self, *args, **kwargs):
            return None

    celery_stub.Celery = _FakeCelery
    sys.modules["celery"] = celery_stub

try:
    from shared.tasks import ocr_tasks
except ModuleNotFoundError:
    ocr_tasks = None


@unittest.skipIf(ocr_tasks is None, "Project dependencies are not installed in current environment")
class TestOCRTasks(unittest.TestCase):
    def setUp(self):
        self.task_self = SimpleNamespace(request=SimpleNamespace(retries=0))

    def test_process_pending_ocr_skips_already_ready(self):
        pending = SimpleNamespace(
            id=10,
            status="ready",
            user_id=1,
            telegram_file_id="abc",
            telegram_chat_id=123,
            mime_type="image/jpeg",
        )
        with patch.object(ocr_tasks, "db_service") as db:
            db.get_pending_document_for_job.return_value = pending
            result = ocr_tasks.process_pending_ocr.run(self.task_self, 10, 1)
            self.assertTrue(result["success"])
            self.assertEqual(result["reason"], "already_processed")

    def test_process_pending_ocr_marks_missing_file_id_failed(self):
        pending = SimpleNamespace(
            id=11,
            status="processing",
            user_id=1,
            telegram_file_id=None,
            telegram_chat_id=123,
            mime_type="image/jpeg",
        )
        with patch.object(ocr_tasks, "db_service") as db, patch.object(ocr_tasks, "_send_telegram_failure") as send_fail:
            db.get_pending_document_for_job.return_value = pending
            result = ocr_tasks.process_pending_ocr.run(self.task_self, 11, 1)
            self.assertFalse(result["success"])
            self.assertEqual(result["reason"], "missing_telegram_file_id")
            db.mark_pending_ocr_failed.assert_called_once()
            send_fail.assert_called_once()

    def test_process_pending_ocr_duplicate_after_ocr(self):
        pending = SimpleNamespace(
            id=12,
            status="processing",
            user_id=7,
            telegram_file_id="tg-file",
            telegram_chat_id=555,
            mime_type="image/jpeg",
        )
        with patch.object(ocr_tasks, "db_service") as db, \
             patch.object(ocr_tasks, "_run") as run, \
             patch.object(ocr_tasks, "_send_telegram_info") as send_info:
            db.get_pending_document_for_job.return_value = pending
            run.return_value = '{"document_type":"invoice"}'
            db.find_duplicate_by_extracted_fingerprint.return_value = {"id": 99}

            result = ocr_tasks.process_pending_ocr.run(self.task_self, 12, 7)
            self.assertFalse(result["success"])
            self.assertEqual(result["reason"], "duplicate_after_ocr")
            db.mark_pending_ocr_failed.assert_called_once()
            send_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
