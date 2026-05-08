import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger("image_hash_service")


def generate_image_hashes(image_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Generate dHash and pHash for image duplicate detection."""
    try:
        from PIL import Image
        import imagehash
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        dhash = str(imagehash.dhash(image))
        phash = str(imagehash.phash(image))
        return dhash, phash
    except ModuleNotFoundError:
        logger.warning("Pillow/ImageHash not installed; duplicate detection is temporarily disabled.")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to generate image hashes: {e}")
        return None, None
