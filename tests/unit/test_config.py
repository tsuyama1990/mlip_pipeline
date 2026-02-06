from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import GlobalConfig


def test_config_valid() -> None:
    config = GlobalConfig(work_dir=Path("/data/work"), max_cycles=10, random_seed=42)
    assert config.work_dir == Path("/data/work")
    assert config.max_cycles == 10


def test_config_invalid_work_dir_empty() -> None:
    with pytest.raises(ValidationError) as exc:
        GlobalConfig(work_dir=Path(), max_cycles=10)
    assert "Work directory cannot be empty or current directory" in str(exc.value)


def test_config_invalid_work_dir_traversal() -> None:
    with pytest.raises(ValidationError) as exc:
        GlobalConfig(work_dir=Path("../secret"), max_cycles=10)
    assert "Work directory path cannot contain '..'" in str(exc.value)


def test_config_invalid_max_cycles() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(
            work_dir=Path("/data"),
            max_cycles=0,  # Should be > 0
        )
