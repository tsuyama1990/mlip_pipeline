from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import OracleConfig


def test_oracle_config_mock_default() -> None:
    """Test that default config is mock and valid."""
    config = OracleConfig()
    assert config.type == "mock"
    assert config.command is None


def test_oracle_config_espresso_valid(tmp_path: Path) -> None:
    """Test that espresso config with all fields is valid."""
    pseudo_dir = tmp_path / "pseudos"
    config = OracleConfig(
        type="espresso",
        command="pw.x",
        pseudo_dir=pseudo_dir,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.05,
        scf_params={"mixing_beta": 0.3},
    )
    assert config.type == "espresso"
    assert config.command == "pw.x"
    assert config.pseudo_dir == pseudo_dir
    assert config.kspacing == 0.05
    assert config.scf_params["mixing_beta"] == 0.3


def test_oracle_config_espresso_missing_command(tmp_path: Path) -> None:
    """Test that espresso config requires command."""
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(
            type="espresso",
            pseudo_dir=tmp_path / "pseudos",
        )
    assert "command is required" in str(excinfo.value)


def test_oracle_config_espresso_missing_pseudo_dir() -> None:
    """Test that espresso config requires pseudo_dir."""
    with pytest.raises(ValidationError) as excinfo:
        OracleConfig(
            type="espresso",
            command="pw.x",
        )
    assert "pseudo_dir is required" in str(excinfo.value)
