from pathlib import Path

from mlip_autopipec.utils.file_ops import atomic_write


def test_atomic_write_success(tmp_path: Path) -> None:
    dest = tmp_path / "success.txt"
    with atomic_write(dest) as temp, temp.open("w") as f:
        f.write("content")

    assert dest.exists()
    assert dest.read_text() == "content"

def test_atomic_write_exception(tmp_path: Path) -> None:
    dest = tmp_path / "fail.txt"

    def _should_raise() -> None:
        with atomic_write(dest) as temp:
            # Manually create the temp file to ensure it exists for cleanup check
            with temp.open("w") as f:
                f.write("partial")
            msg = "Fail"
            raise RuntimeError(msg)

    import contextlib
    with contextlib.suppress(RuntimeError):
        _should_raise()

    assert not dest.exists()
    # Check temp file is cleaned up.
    temp_name = f".{dest.name}.tmp"
    assert not (tmp_path / temp_name).exists()
