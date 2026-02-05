from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import GlobalConfig


def test_valid_global_config() -> None:
    config = GlobalConfig(work_dir=Path("./workspace"), max_cycles=3, random_seed=42)
    assert config.max_cycles == 3
    assert config.work_dir == Path("./workspace")


def test_invalid_max_cycles() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(
            work_dir=Path("./workspace"),
            max_cycles=0,  # Must be > 0
            random_seed=42,
        )
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./workspace"), max_cycles=-1, random_seed=42)


def test_missing_fields() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=3)  # Missing work_dir # type: ignore


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(
            work_dir=Path("./workspace"),
            max_cycles=3,
            extra_field="invalid",
        )
