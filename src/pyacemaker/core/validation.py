"""Validation utilities for PYACEMAKER."""

import re
import tempfile
from pathlib import Path
from typing import Any


def _validate_absolute_path(path: Path, cwd: Path, allow_temp_dirs: bool = True) -> Path:
    """Validate an absolute path against CWD and the allowed-paths whitelist."""
    from pyacemaker.core.config import CONSTANTS

    # Resolve the path (or its parent if it doesn't exist yet).
    # This prevents symlink traversal attacks.
    if path.exists():
        resolved = path.resolve(strict=True).absolute()
    else:
        # If it doesn't exist, validate the parent directory exists and is safe
        try:
            parent = path.parent.resolve(strict=True)
            resolved = (parent / path.name).absolute()
        except FileNotFoundError as e:
            msg = f"Parent directory does not exist for path: {path}"
            raise ValueError(msg) from e

    # Check against CWD (default safe scope)
    if resolved.is_relative_to(cwd):
        return resolved

    # Check against system temp dir if allowed
    if allow_temp_dirs:
        try:
            temp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
            if resolved.is_relative_to(temp_dir):
                return resolved
        except (ValueError, OSError, RuntimeError):
            # If temp dir resolution fails, ignore and proceed to whitelist check
            pass

    # Check against explicit whitelists (e.g. for MACE cache or external datasets)
    for allowed in CONSTANTS.allowed_potential_paths:
        try:
            # Expand ~ if present, though resolve() usually handles relative paths
            allowed_path = Path(allowed).expanduser().resolve(strict=True)
            if resolved.is_relative_to(allowed_path):
                return resolved
        except (OSError, RuntimeError):
            # Skip invalid/non-existent whitelist entries without crashing
            continue

    msg = f"Path {resolved} is not within current working directory or allowed whitelist"
    raise ValueError(msg)


def validate_safe_path(path: Path, allow_temp_dirs: bool = True) -> Path:
    """Validate that path is safe (within CWD or whitelisted).

    This function defends against path traversal attacks by resolving symlinks
    and ensuring the final target is within trusted boundaries.
    """
    from pyacemaker.core.config import CONSTANTS

    if str(path).strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    # Always reject explicit path traversal components in the string representation
    if ".." in path.parts:
        msg = (
            f"Path traversal detected in '{path}': '..' component is forbidden."
        )
        raise ValueError(msg)

    # If security checks are globally disabled (e.g. for testing mocks), bypass logic
    if CONSTANTS.skip_file_checks:
        return path

    try:
        # Establish the trusted base (Project Root or CWD)
        # Note: In a real app, this should be config.project.root_dir, but we default to CWD here.
        cwd = Path.cwd().resolve(strict=True)

        if not path.is_absolute():
            # Convert to absolute path relative to CWD for verification
            candidate = (cwd / path).resolve()

            # Re-verify containment after resolution to catch sneaky symlinks
            # AND validate against whitelist using _validate_absolute_path (Security Fix)
            # This ensures that even if it looks like it's in CWD, the resolved path is also safe
            # and follows all whitelist rules.
            _validate_absolute_path(candidate, cwd, allow_temp_dirs=allow_temp_dirs)
            return path

        # For absolute paths, perform full validation against whitelist
        return _validate_absolute_path(path, cwd, allow_temp_dirs=allow_temp_dirs)

    except (ValueError, RuntimeError, OSError) as e:
        # Wrap all path errors in ValueError for consistent API handling
        msg = f"Path validation failed for {path}: {e}"
        raise ValueError(msg) from e


def validate_parameters(data: dict[str, Any]) -> dict[str, Any]:
    """Validate parameters dictionary against security rules."""
    _validate_structure(data)
    return data


def _validate_structure(data: Any, path: str = "", depth: int = 0) -> None:
    """Validate data structure recursively."""
    from pyacemaker.core.config import CONSTANTS

    # Use constant from config for validation logic
    valid_value_regex = re.compile(CONSTANTS.valid_value_regex)

    if depth > 10:
        msg = "Configuration nesting too deep (max 10)"
        raise ValueError(msg)

    if isinstance(data, dict):
        _validate_dict(data, path, depth)
    elif isinstance(data, (list, tuple)):
        _validate_list(data, path, depth)
    elif isinstance(data, str):
        if not valid_value_regex.match(data):
            msg = f"Invalid characters in value at '{path}': {data}. Must match pattern: {valid_value_regex.pattern}"
            raise ValueError(msg)
    elif not isinstance(data, (int, float, bool, type(None))):
        msg = f"Invalid type {type(data)} at {path}"
        raise TypeError(msg)


def _validate_dict(data: dict[str, Any], path: str, depth: int) -> None:
    from pyacemaker.core.config import CONSTANTS

    valid_key_regex = re.compile(CONSTANTS.valid_key_regex)

    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key
        if not isinstance(key, str):
            msg = f"Keys must be strings at {current_path}"
            raise TypeError(msg)

        if not valid_key_regex.match(key):
            msg = (
                f"Invalid characters in key '{current_path}'. Must match {valid_key_regex.pattern}"
            )
            raise ValueError(msg)

        _validate_structure(value, current_path, depth + 1)


def _validate_list(data: list[Any] | tuple[Any, ...], path: str, depth: int) -> None:
    for i, value in enumerate(data):
        current_path = f"{path}[{i}]"
        _validate_structure(value, current_path, depth + 1)
