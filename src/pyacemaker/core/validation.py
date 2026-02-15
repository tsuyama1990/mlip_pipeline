"""Validation utilities for PYACEMAKER."""

import re
from pathlib import Path
from typing import Any

# Defer import to avoid circular dependency
# from pyacemaker.core.config import CONSTANTS


def validate_safe_path(path: Path) -> Path:
    """Validate that path is safe (within CWD or whitelisted)."""
    from pyacemaker.core.config import CONSTANTS

    if str(path).strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    if CONSTANTS.skip_file_checks:
        return path

    try:
        cwd = Path.cwd().resolve()
        # Strict resolution (resolve symlinks)
        resolved = path.resolve(strict=True) if path.exists() else path.absolute()

        # 1. Containment Check
        if resolved.is_relative_to(cwd):
            return resolved

        # 2. Whitelist Check
        if (
            resolved.is_absolute()
            and resolved.exists()
            and any(
                str(resolved).startswith(str(Path(p).resolve()))
                for p in CONSTANTS.allowed_potential_paths
            )
        ):
            return resolved

        msg = f"Path is outside CWD and not in allowed whitelist: {resolved}"
        raise ValueError(msg)  # noqa: TRY301

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
            msg = f"Invalid characters in value at '{path}'. Found potentially unsafe characters."
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
            msg = f"Invalid characters in key '{current_path}'. Must match {valid_key_regex.pattern}"
            raise ValueError(msg)

        _validate_structure(value, current_path, depth + 1)


def _validate_list(data: list[Any] | tuple[Any, ...], path: str, depth: int) -> None:
    for i, value in enumerate(data):
        current_path = f"{path}[{i}]"
        _validate_structure(value, current_path, depth + 1)
