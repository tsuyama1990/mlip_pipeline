import pytest
from mlip_autopipec.utils.file_ops import atomic_write
from pathlib import Path

def test_atomic_write_success(tmp_path):
    dest = tmp_path / "success.txt"
    with atomic_write(dest) as temp:
        with open(temp, "w") as f:
            f.write("content")

    assert dest.exists()
    assert dest.read_text() == "content"

def test_atomic_write_exception(tmp_path):
    dest = tmp_path / "fail.txt"
    try:
        with atomic_write(dest) as temp:
            # Manually create the temp file to ensure it exists for cleanup check
            with open(temp, "w") as f:
                f.write("partial")
            raise RuntimeError("Fail")
    except RuntimeError:
        pass

    assert not dest.exists()
    # Check temp file is cleaned up.
    temp_name = f".{dest.name}.tmp"
    assert not (tmp_path / temp_name).exists()
