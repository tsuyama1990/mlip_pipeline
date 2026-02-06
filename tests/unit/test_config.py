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
    """
    Tests that GlobalConfig requires mandatory fields and sets defaults correctly.
    """
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
    """
    Tests that GlobalConfig raises ValidationError when required fields (work_dir, random_seed) are missing.
    """
    # Missing work_dir and random_seed
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=5)  # type: ignore[call-arg]


def test_global_config_valid(tmp_path: Path) -> None:
    """
    Tests that a fully specified GlobalConfig is valid.
    """
    work_dir = tmp_path / "test"
    config = GlobalConfig(
        work_dir=work_dir,
        max_cycles=10,
        random_seed=123,
        explorer=ExplorerConfig(type="random"),
        oracle=OracleConfig(
            type="espresso",
            command="pw.x",
            pseudo_dir=work_dir / "pseudos",
            pseudopotentials={"Si": "Si.UPF"},
        ),
        trainer=TrainerConfig(type="pacemaker"),
        validator=ValidatorConfig(type="mock"),
    )
    assert config.work_dir == work_dir
    assert config.max_cycles == 10
    assert config.explorer.type == "random"


def test_global_config_invalid_max_cycles(tmp_path: Path) -> None:
    """
    Tests that max_cycles must be >= 1.
    """
    work_dir = tmp_path / "test"
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=work_dir, max_cycles=0, random_seed=42)  # Must be >= 1


def test_global_config_extra_forbid(tmp_path: Path) -> None:
    """
    Tests that extra fields are forbidden in GlobalConfig.
    """
    work_dir = tmp_path / "test"
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=work_dir, max_cycles=5, random_seed=42, extra_field="forbidden")  # type: ignore[call-arg]


def test_trainer_config_invalid_potential_name() -> None:
    """
    Tests that TrainerConfig validates the potential output name extension.
    """
    with pytest.raises(ValidationError):
        TrainerConfig(potential_output_name="bad_name.txt")


def test_trainer_config_valid_potential_name() -> None:
    """
    Tests that a valid potential output name is accepted.
    """
    config = TrainerConfig(potential_output_name="good.yace")
    assert config.potential_output_name == "good.yace"


def test_oracle_config_espresso_requires_fields(tmp_path: Path) -> None:
    """Tests that Espresso oracle requires command, pseudo_dir, and pseudopotentials."""
    # Missing all (fails fast on command)
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="espresso")
    assert "command is required" in str(excinfo.value)

    # Missing pseudo_dir
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="espresso", command="pw.x")
    assert "pseudo_dir is required" in str(excinfo.value)

    # Missing pseudopotentials
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="espresso", command="pw.x", pseudo_dir=tmp_path)
    assert "pseudopotentials are required" in str(excinfo.value)


def test_oracle_config_security() -> None:
    """Tests that dangerous commands are rejected."""
    dangerous_commands = ["pw.x > output", "pw.x | grep error", "pw.x; rm -rf /"]
    for cmd in dangerous_commands:
        with pytest.raises(ValidationError):
            OracleConfig(type="espresso", command=cmd)


def test_oracle_config_kspacing() -> None:
    """Tests that kspacing must be positive."""
    with pytest.raises(ValidationError):
        OracleConfig(kspacing=0.0)
    with pytest.raises(ValidationError):
        OracleConfig(kspacing=-0.1)
