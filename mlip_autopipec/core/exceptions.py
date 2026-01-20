"""
Custom exceptions for the MLIP-AutoPipe project.
"""

class MLIPError(Exception):
    """Base exception for all MLIP-AutoPipe errors."""

class ConfigError(MLIPError):
    """Raised when configuration validation fails."""

class DFTError(MLIPError):
    """Base exception for DFT-related errors."""

class DFTConvergenceError(DFTError):
    """Raised when a DFT calculation fails to converge."""

class DFTRuntimeError(DFTError):
    """Raised when the DFT binary fails to run or crashes."""
