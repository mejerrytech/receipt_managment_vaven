from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_FORMATS = ("pdf", "csv", "both")
