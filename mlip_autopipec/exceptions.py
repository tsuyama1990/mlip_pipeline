class MLIPError(Exception):
    """Base exception for MLIP-AutoPipe."""

class ConfigError(MLIPError):
    """Raised when configuration loading or validation fails."""

class WorkspaceError(MLIPError):
    """Raised when workspace setup fails (e.g., directory creation)."""

class DatabaseError(MLIPError):
    """Raised when database operations fail."""

class LoggingError(MLIPError):
    """Raised when logging setup fails."""
