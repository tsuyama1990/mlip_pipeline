"""Validation utilities for PYACEMAKER."""

import re
from pathlib import Path
from typing import Any


def _validate_absolute_path(path: Path, cwd: Path) -> Path:
    """Validate an absolute path against CWD and the allowed-paths whitelist."""
    from pyacemaker.core.config import CONSTANTS

    # Resolve the path (or its parent if it doesn't exist yet).
    if path.exists():
        resolved = path.resolve(strict=True).absolute()
    else:
        parent = path.parent.resolve(strict=True)
        resolved = (parent / path.name).absolute()

    if resolved.is_relative_to(cwd):
        return resolved

    for allowed in CONSTANTS.allowed_potential_paths:
        allowed_path = Path(allowed).resolve(strict=True)
        if resolved.is_relative_to(allowed_path):
            return resolved

    msg = f"Path {resolved} is not within current working directory or allowed whitelist"
    raise ValueError(msg)


def validate_safe_path(path: Path) -> Path:
    """Validate that path is safe (within CWD or whitelisted)."""
    from pyacemaker.core.config import CONSTANTS

    if str(path).strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    # Always reject explicit path traversal components, regardless of skip_file_checks.
    if ".." in path.parts:
        msg = (
            f"Path traversal not allowed: {path} must be strictly within current working directory"
        )
        raise ValueError(msg)

    if CONSTANTS.skip_file_checks:
        return path

    try:
        cwd = Path.cwd().resolve(strict=True)

        # For relative paths: check containment lexically without requiring existence on disk.
        if not path.is_absolute():
            candidate = (cwd / path).resolve()
            if candidate.is_relative_to(cwd):
                return path
            msg = f"Path {path} must be strictly within current working directory"
            raise ValueError(msg)  # noqa: TRY301

        return _validate_absolute_path(path, cwd)

    except (ValueError, RuntimeError, OSError) as e:
        msg = f"Path {path} is unsafe or outside allowed base directory: {e}"
        raise ValueError(msg) from e


def validate_parameters(data: dict[str, Any]) -> dict[str, Any]:
    """Validate parameters dictionary against security rules."""
    _validate_structure(data)
    return data


def _validate_structure(data: Any, path: str = "", depth: int = 0) -> None:
    """Validate data structure recursively."""
    from pyacemaker.core.config import CONSTANTS

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
