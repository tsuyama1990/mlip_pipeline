class PyAceError(Exception):
    """Base exception for PyAceMaker."""


class ConfigurationError(PyAceError):
    """Invalid configuration."""


class ComponentError(PyAceError):
    """Error in a component execution."""


class StateError(PyAceError):
    """Error in state management."""


class OrchestratorError(PyAceError):
    """Error in orchestration logic."""
