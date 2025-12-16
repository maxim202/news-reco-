import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    #logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)
