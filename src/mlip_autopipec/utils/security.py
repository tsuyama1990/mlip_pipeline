from pathlib import Path


def validate_safe_path(path: Path, base_dir: Path | None = None) -> Path:
    """
    Validate that a path is safe to use (no traversal, strict existence check if needed).

    Args:
        path: The path to validate.
        base_dir: Optional base directory to confine the path to.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If path is unsafe or traversal is detected.
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

    return resolved
