from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig, OracleConfig


def test_valid_config() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        "workdir": "/tmp/test_project",  # noqa: S108
        "seed": 42,
        "max_cycles": 5,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig(**data)
    assert config.project_name == "test_project"
    assert config.oracle.type == "mock"
    assert str(config.workdir) == "/tmp/test_project" # noqa: S108


def test_missing_oracle() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        "workdir": "/tmp/test_project", # noqa: S108
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)

def test_oracle_config_defaults() -> None:
    # Test defaults for mock
    config = OracleConfig(type="mock")
    assert config.type == "mock"
    assert config.kspacing == 0.04
    assert config.smearing_width == 0.02
    assert config.command is None

def test_oracle_config_qe_valid() -> None:
    # Test valid QE config
    data = {
        "type": "qe",
        "command": "pw.x",
        "pseudo_dir": "/tmp/pseudos", # noqa: S108
        "pseudopotentials": {"Si": "Si.upf"},
        "kspacing": 0.05,
        "smearing_width": 0.01
    }
    config = OracleConfig(**data)
    assert config.type == "qe"
    assert config.command == "pw.x"
    assert config.pseudo_dir == Path("/tmp/pseudos") # noqa: S108
    assert config.pseudopotentials == {"Si": "Si.upf"}
    assert config.kspacing == 0.05
    assert config.smearing_width == 0.01

def test_oracle_config_qe_missing_fields() -> None:
    # Test missing fields for QE
    # Missing command
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="qe", pseudo_dir=Path("/tmp"), pseudopotentials={}) # noqa: S108
    assert "QE Oracle requires 'command'" in str(excinfo.value)

    # Missing pseudo_dir
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="qe", command="pw.x", pseudopotentials={})
    assert "QE Oracle requires 'pseudo_dir'" in str(excinfo.value)

    # Missing pseudopotentials
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="qe", command="pw.x", pseudo_dir=Path("/tmp")) # noqa: S108
    assert "QE Oracle requires 'pseudopotentials'" in str(excinfo.value)

def test_oracle_config_invalid_values() -> None:
    # Test invalid kspacing
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="mock", kspacing=-0.1)
    assert "kspacing must be positive" in str(excinfo.value)

    # Test invalid smearing_width
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(type="mock", smearing_width=0.0)
    assert "smearing_width must be positive" in str(excinfo.value)
