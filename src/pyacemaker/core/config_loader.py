"""Configuration loading and I/O utilities.

Handles secure file reading, validation, and parsing.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.exceptions import ConfigurationError


def _check_file_size(file_size: int) -> None:
    """Check if file size exceeds limit."""
    if file_size > CONSTANTS.max_config_size:
        msg = f"Configuration file too large: {file_size} bytes (max {CONSTANTS.max_config_size})"
        raise ConfigurationError(msg)


def _validate_file_security(path: Path) -> None:
    """Validate file permissions and ownership."""
    if CONSTANTS.skip_file_checks:
        return

    # Resolve symlinks to check actual file
    real_path = path.resolve()
    if not real_path.is_file():
        msg = f"Path is not a regular file: {path.name}"
        raise ConfigurationError(msg)

    # Security: Disallow symlinks completely to prevent attacks
    if path.is_symlink():
        msg = f"Configuration file cannot be a symlink: {path.name}"
        raise ConfigurationError(msg)

    # Security: Ensure path is within CWD or allowed base
    try:
        cwd = Path.cwd().resolve()
        if not real_path.is_relative_to(cwd):
            msg = f"Configuration file must be within current working directory: {cwd}"
            raise ConfigurationError(msg)
    except ValueError as e:
        msg = f"Configuration file path {real_path} is outside allowed base directory {cwd}"
        raise ConfigurationError(msg) from e

    # Check file permissions (Security)
    if not os.access(path, os.R_OK):
        msg = f"Permission denied: {path.name}"
        raise ConfigurationError(msg)

    try:
        st = real_path.stat()
        # Check for world-writable (0o002) AND group-writable (0o020)
        # Group writable is risky if shared environment
        if st.st_mode & (0o002 | 0o020):
             msg = f"Configuration file {path.name} is world/group-writable. This is insecure."
             raise ConfigurationError(msg)

        # Check for executable (config files should not be executable)
        if st.st_mode & 0o111:
            msg = f"Configuration file {path.name} is executable. This is insecure."
            raise ConfigurationError(msg)

        # Check ownership: Must be owned by current user
        if st.st_uid != os.getuid():
             msg = f"Configuration file {path.name} is not owned by the current user."
             raise ConfigurationError(msg)

    except OSError as e:
        msg = f"Error checking file permissions: {e}"
        raise ConfigurationError(msg) from e


def _read_config_file(path: Path) -> dict[str, Any]:
    """Read and parse configuration file safely.

    Uses stream parsing to avoid loading entire file into memory string,
    while enforcing size limits.
    """
    try:
        file_size = path.stat().st_size
        _check_file_size(file_size)
    except OSError as e:
        msg = f"Error accessing configuration file: {e}"
        raise ConfigurationError(msg, details={"filename": path.name}) from e

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration: {e}"
        raise ConfigurationError(msg, details={"original_error": str(e)}) from e
    except Exception as e:
        # Catch generic errors to ensure consistent exception wrapping
        msg = f"Unexpected error reading configuration: {e}"
        raise ConfigurationError(msg) from e
    else:
        if not isinstance(data, dict):
            msg = f"Configuration file must contain a YAML dictionary, got {type(data).__name__}."
            raise ConfigurationError(msg)

        # Validate top-level keys against schema to prevent injection of unknown sections
        # before Pydantic sees it (Defensive)
        allowed_keys = set(PYACEMAKERConfig.model_fields.keys())
        # We can also allow comments or extra keys if needed, but strict is better.
        # Pydantic is configured with extra='forbid', so it will catch it too.
        # But this explicit check satisfies "Validate structure before Pydantic".
        unknown_keys = set(data.keys()) - allowed_keys
        if unknown_keys:
             msg = f"Unknown top-level configuration keys: {unknown_keys}"
             raise ConfigurationError(msg)

        return data


def load_config(path: Path) -> PYACEMAKERConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PYACEMAKERConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.

    """
    if not path.exists():
        msg = f"Configuration file not found: {path.name}"
        raise ConfigurationError(msg)

    _validate_file_security(path)

    try:
        data = _read_config_file(path)
        return PYACEMAKERConfig(**data)
    except ConfigurationError:
        raise
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        details = {"errors": e.errors()}
        raise ConfigurationError(msg, details=details) from e
    except Exception as e:
        # Catch unexpected errors during load
        self_logger = __import__("logging").getLogger("pyacemaker.core.config_loader")
        self_logger.exception("Unexpected error loading configuration")
        msg = f"Unexpected error loading configuration: {e}"
        raise ConfigurationError(msg) from e
