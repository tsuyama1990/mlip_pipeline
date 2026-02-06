from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config import (
    ExplorerConfig,
    GlobalConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)


def test_global_config_defaults_validation() -> None:
    # Must provide work_dir and random_seed
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=5) # type: ignore[call-arg]

    config = GlobalConfig(max_cycles=5, work_dir=Path("./_work"), random_seed=42)
    assert config.max_cycles == 5
    assert config.work_dir == Path("./_work")
    assert config.random_seed == 42
    assert config.explorer.type == "mock"
    assert config.oracle.type == "mock"
    assert config.trainer.type == "mock"
    assert config.validator.type == "mock"

def test_global_config_valid(tmp_path: Path) -> None:
    work_dir = tmp_path / "test"
    config = GlobalConfig(
        work_dir=work_dir,
        max_cycles=10,
        random_seed=123,
        explorer=ExplorerConfig(type="random"),
        oracle=OracleConfig(type="espresso"),
        trainer=TrainerConfig(type="pacemaker"),
        validator=ValidatorConfig(type="custom_type") # Now allows str
    )
    assert config.work_dir == work_dir
    assert config.max_cycles == 10
    assert config.explorer.type == "random"

def test_global_config_invalid_max_cycles() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=0, work_dir=Path(), random_seed=1)  # Must be >= 1

def test_global_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=5, work_dir=Path(), random_seed=1, extra_field="forbidden") # type: ignore[call-arg]
