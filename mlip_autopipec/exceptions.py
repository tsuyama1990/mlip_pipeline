"""
This module defines custom exceptions for the MLIP-AutoPipe application.

Using custom exceptions allows for more specific and expressive error
handling throughout the workflow.
"""

from typing import Any


class MLIPException(Exception):
    """Base exception for all MLIP-AutoPipe errors."""


class ConfigException(MLIPException):
    """Raised when configuration loading or validation fails."""


class DatabaseException(MLIPException):
    """Raised when database operations fail."""


class WorkspaceException(MLIPException):
    """Raised when workspace setup or filesystem operations fail."""


class LoggingException(MLIPException):
    """Raised when logging setup fails."""


class DFTException(MLIPException):
    """Base exception for DFT operations."""


class DFTCalculationException(DFTException):
    """
    Raised when a DFT calculation fails and cannot be recovered automatically.
    """

    def __init__(
        self,
        message: str,
        stdout: str = "",
        stderr: str = "",
        is_timeout: bool = False,
    ) -> None:
        super().__init__(message)
        # Truncate stdout/stderr if too long
        max_len = 5000
        self.stdout = stdout if len(stdout) <= max_len else stdout[:max_len] + "... [TRUNCATED]"
        self.stderr = stderr if len(stderr) <= max_len else stderr[:max_len] + "... [TRUNCATED]"
        self.is_timeout = is_timeout

    def __str__(self) -> str:
        timeout_msg = "[TIMEOUT] " if self.is_timeout else ""
        return f"{timeout_msg}{super().__str__()}\n--- STDOUT ---\n{self.stdout}\n--- STDERR ---\n{self.stderr}"


class GeneratorException(MLIPException):
    """Raised when structure generation fails."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.context:
            return f"{base_msg} | Context: {self.context}"
        return base_msg
