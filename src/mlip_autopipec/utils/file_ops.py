from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_write(filepath: Path) -> Generator[Path, None, None]:
    """
    Context manager for atomic file writing.
    Writes to a temporary file first, then renames it to the destination.

    Usage:
        with atomic_write(final_path) as temp_path:
            with open(temp_path, "w") as f:
                f.write(data)
    """
    filepath = Path(filepath)
    parent = filepath.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_name = f".{filepath.name}.tmp"
    temp_path = parent / temp_name

    try:
        yield temp_path
        temp_path.replace(filepath)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
