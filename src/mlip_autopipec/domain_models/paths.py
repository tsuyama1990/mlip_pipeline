from pathlib import Path


def validate_path_safety(v: Path | None) -> Path | None:
    """
    Validates that a path is safe (no traversal).

    This function checks if the path contains '..' components in its parts,
    which would indicate a directory traversal attempt.
    Absolute paths are allowed (as users may specify absolute paths for inputs),
    but relative paths must not attempt to go up the directory tree.
    """
    if v is None:
        return None

    # Check for explicit traversal components
    if ".." in v.parts:
        msg = f"Path traversal detected in {v}"
        raise ValueError(msg)

    return v
