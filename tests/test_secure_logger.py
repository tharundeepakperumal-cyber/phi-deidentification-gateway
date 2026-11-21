
import logging

from src.secure_logger import (
    SensitiveDataFilter,
    setup_secure_logging,
)


def test_sensitive_data_filter():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="Patient email test@example.com, phone +919876543210, MRN: ABC123",
        args=(),
        exc_info=None,
    )

    phi_filter = SensitiveDataFilter()
    assert phi_filter.filter(record)

    assert "test@example.com" not in record.getMessage()
    assert "+919876543210" not in record.getMessage()
    assert "ABC123" not in record.getMessage()


def test_secure_logger_setup():
    logger = setup_secure_logging()

    assert logger.name == "phi_gateway"
    assert logger.level == logging.INFO
