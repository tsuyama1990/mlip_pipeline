from pathlib import Path

from mlip_autopipec.utils.file_ops import atomic_write


def test_atomic_write_success(tmp_path: Path) -> None:
    dest = tmp_path / "success.txt"
    with atomic_write(dest) as temp, temp.open("w") as f:
        f.write("content")

    assert dest.exists()
    assert dest.read_text() == "content"


def test_atomic_write_failure(tmp_path: Path) -> None:
    dest = tmp_path / "fail.txt"
    try:
        with atomic_write(dest) as temp:
            # Manually create the temp file to ensure it exists for cleanup check
            with temp.open("w") as f:
                f.write("partial")
            msg = "Fail"
            raise RuntimeError(msg)  # noqa: TRY301
    except RuntimeError:
        pass

    assert not dest.exists()
    # Temp file should also be gone
    assert not any(tmp_path.iterdir())
