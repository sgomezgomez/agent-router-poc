"""Structured logging for agent router."""

import logging
import json
from datetime import datetime
from agent_router.core.types import JsonObject
from agent_router.core.config import Settings


class StructuredLogger:
    """Structured JSON logger for the agent router."""

    def __init__(self, name: str, settings: Settings):
        self.logger = logging.getLogger(name)
        self.settings = settings

        # Configure based on settings
        level = getattr(logging, settings.logging.level.upper())
        self.logger.setLevel(level)

        # Create handler
        if settings.logging.file:
            handler = logging.FileHandler(settings.logging.file)
        else:
            handler = logging.StreamHandler()

        # Format as JSON if configured
        if settings.logging.format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )

        self.logger.addHandler(handler)

    def info(self, message: str, **kwargs):
        """Log info level message with structured data."""
        self._log(logging.INFO, message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message with structured data."""
        self._log(logging.ERROR, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level message with structured data."""
        self._log(logging.WARNING, message, kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message with structured data."""
        self._log(logging.DEBUG, message, kwargs)

    def _log(self, level: int, message: str, extra: JsonObject):
        """Internal logging method."""
        self.logger.log(level, message, extra={"structured_data": extra})


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured data if present
        if hasattr(record, "structured_data"):
            log_data.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)
