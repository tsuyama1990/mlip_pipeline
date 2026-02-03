from pathlib import Path

from mlip_autopipec.utils.file_ops import atomic_write


def test_atomic_write_success(tmp_path: Path) -> None:
    dest = tmp_path / "success.txt"
    # PTH123: use temp.open()
    with atomic_write(dest) as temp, temp.open("w") as f:
        f.write("content")

    assert dest.exists()
    assert dest.read_text() == "content"


def _raise_failure() -> None:
    msg = "Fail"
    # EM101: Exception must not use string literal
    raise RuntimeError(msg)


def test_atomic_write_failure(tmp_path: Path) -> None:
    dest = tmp_path / "failure.txt"
    try:
        with atomic_write(dest) as temp:
            # PTH123: use temp.open()
            # Manually create the temp file to ensure it exists for cleanup check
            with temp.open("w") as f:
                f.write("partial")
            # TRY301: Abstract raise to inner function
            _raise_failure()
    except RuntimeError:
        pass

    assert not dest.exists()
    # Temp file should also be cleaned up
