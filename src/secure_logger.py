
import logging
import re


class SensitiveDataFilter(logging.Filter):
    """Remove common PHI patterns from log messages."""

    PATTERNS = [
        (r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]'),
        (r'\b\+?\d[\d\s()-]{8,}\d\b', '[PHONE_REDACTED]'),
        (r'\bMRN[-:\s]*[A-Za-z0-9-]+\b', '[MRN_REDACTED]'),
    ]

    def filter(self, record):
        message = record.getMessage()

        for pattern, replacement in self.PATTERNS:
            message = re.sub(
                pattern,
                replacement,
                message,
                flags=re.IGNORECASE
            )

        record.msg = message
        record.args = ()

        return True


def setup_secure_logging():
    """Configure application logging with PHI filtering."""

    logger = logging.getLogger("phi_gateway")
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(f, SensitiveDataFilter)
        for f in logger.filters
    ):
        logger.addFilter(SensitiveDataFilter())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

    return logger
