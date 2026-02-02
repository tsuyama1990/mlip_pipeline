from pathlib import Path

import pytest

from mlip_autopipec.utils.file_ops import atomic_write


def test_atomic_write_success(temp_dir: Path) -> None:
    target = temp_dir / "target.txt"
    with atomic_write(target) as temp:
        temp.write_text("success")
        assert temp.exists()
        assert not target.exists()

    assert target.exists()
    assert target.read_text() == "success"


def test_atomic_write_failure(temp_dir: Path) -> None:
    target = temp_dir / "target.txt"

    with pytest.raises(RuntimeError):  # noqa: PT012, SIM117
        with atomic_write(target) as temp:
            temp.write_text("fail")
            raise RuntimeError("Boom")  # noqa: EM101

    assert not target.exists()
    # Ensure temp file is cleaned up
    # The temp file name logic is target.name + ".tmp"
    temp_file = target.with_name(target.name + ".tmp")
    assert not temp_file.exists()
