from pathlib import Path


def validate_safe_path(path: Path, base_dir: Path | None = None, must_exist: bool = False) -> Path:
    """
    Validate that a path is safe to use (no traversal, strict existence check if needed).

    Args:
        path: The path to validate.
        base_dir: Optional base directory to confine the path to.
        must_exist: If True, the path must exist.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If path is unsafe or traversal is detected.
        FileNotFoundError: If must_exist is True and path does not exist.
    """
    try:
        # Resolve strictly
        resolved = path.resolve()
    except OSError as e:
        msg = f"Invalid path: {path}"
        raise ValueError(msg) from e

    # Check for null bytes
    if "\0" in str(resolved):
        msg = "Path contains null bytes"
        raise ValueError(msg)

    if base_dir:
        try:
            resolved_base = base_dir.resolve()
            if not resolved.is_relative_to(resolved_base):
                msg = f"Path {resolved} is outside permitted directory {resolved_base}"
                raise ValueError(msg)
        except OSError as e:
            msg = f"Invalid base directory: {base_dir}"
            raise ValueError(msg) from e

    if must_exist and not resolved.exists():
        msg = f"Path does not exist: {resolved}"
        raise FileNotFoundError(msg)

    return resolved
