import logging
import os
from typing import Optional

def setup_logging(level: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
    )
