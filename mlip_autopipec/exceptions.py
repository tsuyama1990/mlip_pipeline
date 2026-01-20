"""
This module defines custom exceptions for the MLIP-AutoPipe application.

Using custom exceptions allows for more specific and expressive error
handling throughout the workflow.
"""

from typing import Any


class MLIPError(Exception):
    """Base exception for all MLIP-AutoPipe errors."""


class ConfigError(MLIPError):
    """Raised when configuration loading or validation fails."""


class DatabaseError(MLIPError):
    """Raised when database operations fail."""


class WorkspaceError(MLIPError):
    """Raised when workspace setup or filesystem operations fail."""


class LoggingError(MLIPError):
    """Raised when logging setup fails."""


class DFTError(MLIPError):
    """Base exception for DFT operations."""


class DFTCalculationError(DFTError):
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


class GeneratorError(MLIPError):
    """Raised when structure generation fails."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.context:
            return f"{base_msg} | Context: {self.context}"
        return base_msg
