from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.config_model import GlobalConfig


def test_valid_config() -> None:
    config = GlobalConfig(work_dir=Path("./test"), max_cycles=1, random_seed=42)
    assert config.max_cycles == 1
    assert str(config.work_dir) == "test"


def test_invalid_max_cycles() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./test"), max_cycles=0)


def test_missing_fields() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./test"))  # type: ignore[call-arg]
