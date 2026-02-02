from pathlib import Path

from mlip_autopipec.utils.file_ops import atomic_write


def test_atomic_write(temp_dir: Path) -> None:
    target_file = temp_dir / "test.txt"

    with atomic_write(target_file) as temp_path:
        # Check temp path is different
        assert temp_path != target_file
        # Check temp path is in same directory
        assert temp_path.parent == target_file.parent
        # Check temp path has .tmp suffix
        assert temp_path.name.endswith(".tmp")

        # Write to temp path
        with temp_path.open("w") as f:
            f.write("hello")

        # Target file should not exist yet
        assert not target_file.exists()

    # After context manager exit, target file should exist
    assert target_file.exists()
    with target_file.open("r") as f:
        assert f.read() == "hello"


def test_atomic_write_failure(temp_dir: Path) -> None:
    target_file = temp_dir / "fail.txt"

    try:
        with atomic_write(target_file) as temp_path:
            with temp_path.open("w") as f:
                f.write("partial")
            raise RuntimeError("Something went wrong")
    except RuntimeError:
        pass

    # Target file should not exist
    assert not target_file.exists()
    # Temp file should be cleaned up
    # Since atomic_write uses .tmp, check for any .tmp file
    assert not any(temp_dir.glob("*.tmp"))
