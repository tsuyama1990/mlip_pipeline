class MLIPError(Exception):
    """Base exception for MLIP-AutoPipe"""


class ConfigError(MLIPError):
    """Configuration validation errors"""


class DFTError(MLIPError):
    """Base class for DFT execution errors"""


class DFTRuntimeError(DFTError):
    """Runtime failure of DFT code (e.g. crash)"""


class DFTConvergenceError(DFTError):
    """SCF convergence failure"""
