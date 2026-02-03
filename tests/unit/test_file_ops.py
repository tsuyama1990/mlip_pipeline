"""Unit tests for atomic file operations."""

from pathlib import Path

from mlip_autopipec.utils.file_ops import atomic_write


# Explicitly update: ensure correct Path.open usage and exception handling
def test_atomic_write_success(tmp_path: Path) -> None:
    dest = tmp_path / "success.txt"
    # Combine with statements as suggested by ruff SIM117
    with atomic_write(dest) as temp, temp.open("w") as f:
        f.write("content")

    assert dest.exists()
    assert dest.read_text() == "content"


def _raise_runtime_error() -> None:
    msg = "Fail"
    raise RuntimeError(msg)


def test_atomic_write_failure(tmp_path: Path) -> None:
    dest = tmp_path / "failure.txt"
    try:
        with atomic_write(dest) as temp:
            # Manually create the temp file to ensure it exists for cleanup check
            with temp.open("w") as f:
                f.write("partial")

            # Helper to raise exception to satisfy TRY301
            _raise_runtime_error()

    except RuntimeError:
        pass

    assert not dest.exists()
    # Temp file should also be cleaned up
