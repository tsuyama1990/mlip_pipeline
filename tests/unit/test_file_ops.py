from pathlib import Path

import pytest

from mlip_autopipec.utils.file_ops import atomic_write


def test_atomic_write_success(temp_dir: Path) -> None:
    target = temp_dir / "target.txt"
    with atomic_write(target) as temp, temp.open("w") as f:
        f.write("success")

    assert temp.exists() is False
    assert target.exists()
    assert target.read_text() == "success"


def _trigger_error() -> None:
    msg = "Boom"
    raise RuntimeError(msg)


def test_atomic_write_failure(temp_dir: Path) -> None:
    target = temp_dir / "target.txt"

    # We suppress PT012 because we need to check cleanup logic which implies wrapping the block.
    # We suppress SIM117 because merging them triggers PT012 or issues with complex context management testing.
    with pytest.raises(RuntimeError):  # noqa: PT012, SIM117
        with atomic_write(target) as temp:
            temp.write_text("fail")
            _trigger_error()

    assert not target.exists()
    # Ensure temp file is cleaned up
    # The temp file name logic is target.name + ".tmp"
    temp_file = target.with_name(target.name + ".tmp")
    assert not temp_file.exists()
