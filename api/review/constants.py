from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "review_images"

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
REVIEW_STATUSES = ("pending", "processing", "ready")
