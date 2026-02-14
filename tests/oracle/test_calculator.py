"""Tests for Calculator Factory."""

from pathlib import Path

import pytest

from pyacemaker.core.config import DFTConfig
from pyacemaker.oracle.calculator import create_calculator


@pytest.fixture
def config(tmp_path: Path) -> DFTConfig:
    """Return a default DFT configuration."""
    pp_file = tmp_path / "H.pbe.UPF"
    pp_file.touch()
    return DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotentials={"H": str(pp_file)},
        kspacing=0.1,
        smearing=0.1,
        max_retries=3,
    )


def test_create_calculator(config: DFTConfig) -> None:
    """Test creating a calculator with default config."""
    calc = create_calculator(config)
    # Check command
    assert calc.profile.command == "pw.x"  # type: ignore[attr-defined]
    assert calc.parameters["kspacing"] == 0.1

    # Check mixing beta default
    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.7


def test_create_calculator_retry(config: DFTConfig) -> None:
    """Test calculator parameters adjustment on retry."""
    calc = create_calculator(config, attempt=1)
    # Check mixing beta adjustment (0.7 - 0.1 = 0.6)
    assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.6


def test_create_calculator_user_override(config: DFTConfig) -> None:
    """Test user parameters override."""
    # User wants to set ecutwfc to 80.0
    config.parameters = {"system": {"ecutwfc": 80.0}}

    calc = create_calculator(config)
    assert calc.parameters["input_data"]["system"]["ecutwfc"] == 80.0
    # Other defaults should remain
    assert calc.parameters["input_data"]["system"]["ecutrho"] == 200.0


def test_create_calculator_unsupported_code(config: DFTConfig) -> None:
    """Test error on unsupported DFT code."""
    # We cheat to bypass validation for this test
    # (Since we validate code in config, but calculator also checks)
    object.__setattr__(config, "code", "vasp")

    with pytest.raises(NotImplementedError, match="not supported"):
        create_calculator(config)
