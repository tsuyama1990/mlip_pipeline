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


def test_global_config_required_fields(tmp_path: Path) -> None:
    work_dir = tmp_path / "test"
    config = GlobalConfig(work_dir=work_dir, max_cycles=5, random_seed=42)
    assert config.max_cycles == 5
    assert config.work_dir == work_dir
    assert config.random_seed == 42
    assert config.max_accumulated_structures == 10000
    assert config.explorer.type == "mock"
    assert config.oracle.type == "mock"
    assert config.trainer.type == "mock"
    assert config.validator.type == "mock"


def test_global_config_defaults_missing() -> None:
    # Missing work_dir and random_seed
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=5)  # type: ignore[call-arg]


def test_global_config_valid(tmp_path: Path) -> None:
    work_dir = tmp_path / "test"
    config = GlobalConfig(
        work_dir=work_dir,
        max_cycles=10,
        random_seed=123,
        explorer=ExplorerConfig(type="random"),
        oracle=OracleConfig(type="espresso"),
        trainer=TrainerConfig(type="pacemaker"),
        validator=ValidatorConfig(type="mock"),
    )
    assert config.work_dir == work_dir
    assert config.max_cycles == 10
    assert config.explorer.type == "random"


def test_global_config_invalid_max_cycles(tmp_path: Path) -> None:
    work_dir = tmp_path / "test"
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=work_dir, max_cycles=0, random_seed=42)  # Must be >= 1


def test_global_config_extra_forbid(tmp_path: Path) -> None:
    work_dir = tmp_path / "test"
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=work_dir, max_cycles=5, random_seed=42, extra_field="forbidden")  # type: ignore[call-arg]


def test_trainer_config_invalid_potential_name() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(potential_output_name="bad_name.txt")


def test_trainer_config_valid_potential_name() -> None:
    config = TrainerConfig(potential_output_name="good.yace")
    assert config.potential_output_name == "good.yace"
