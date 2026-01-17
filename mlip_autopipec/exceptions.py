"""
This module defines custom exceptions for the MLIP-AutoPipe application.

Using custom exceptions allows for more specific and expressive error
handling throughout the workflow.
"""

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

class DFTCalculationError(MLIPError):
    """
    Raised when a DFT calculation fails and cannot be recovered automatically.
    """
    def __init__(
        self,
        message: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return f"{super().__str__()}\n--- STDOUT ---\n{self.stdout}\n--- STDERR ---\n{self.stderr}"
