"""Custom exceptions for the PYACEMAKER application."""


class PYACEMAKERError(Exception):
    """Base exception for all PYACEMAKER errors."""


class ConfigurationError(PYACEMAKERError):
    """Raised when there is an issue with the configuration."""
