from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_write(filepath: Path) -> Generator[Path, None, None]:
    """
    Context manager for atomic file writing.
    Yields a temporary path. On successful exit, renames the temporary path to the target filepath.

    Args:
        filepath: The target file path.
    """
    temp_path = filepath.with_name(filepath.name + ".tmp")
    try:
        yield temp_path
        temp_path.replace(filepath)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
