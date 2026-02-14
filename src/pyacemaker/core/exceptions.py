"""Custom exceptions for the PYACEMAKER application."""

from typing import Any


class PYACEMAKERError(Exception):
    """Base exception for all PYACEMAKER errors."""


class ConfigurationError(PYACEMAKERError):
    """Raised when there is an issue with the configuration.

    Attributes:
        message: Explanation of the error.
        details: Optional dictionary containing additional context (e.g., field names, values).

    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize ConfigurationError."""
        super().__init__(message)
        self.details = details or {}
