import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT

def configure_logger() -> logging.Logger:
    """
    Configure and return the application logger.
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)

    logger = logging.getLogger("HospitalAssistant")

    return logger

logger = configure_logger()