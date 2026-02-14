"""Tests for Calculator Factory."""

from pathlib import Path
from typing import cast

import pytest
from ase.calculators.espresso import Espresso

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
        pseudo_dir=tmp_path,
    )


def test_create_calculator(config: DFTConfig) -> None:
    """Test creating a calculator with default config."""
    calc = create_calculator(config)
    # Cast to Espresso to access specific attributes
    espresso_calc = cast(Espresso, calc)

    # Check command
    # profile is an attribute of Espresso
    assert espresso_calc.profile.command == "pw.x"
    assert espresso_calc.parameters["kspacing"] == 0.1

    # Check mixing beta default
    assert espresso_calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.7


def test_create_calculator_retry(config: DFTConfig) -> None:
    """Test calculator parameters adjustment on retry."""
    calc = create_calculator(config, attempt=1)
    espresso_calc = cast(Espresso, calc)
    # Check mixing beta adjustment (0.7 - 0.1 = 0.6)
    assert espresso_calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.6


def test_create_calculator_user_override(config: DFTConfig) -> None:
    """Test user parameters override."""
    # User wants to set ecutwfc to 80.0
    # Must use allowed sections
    config.parameters = {"system": {"ecutwfc": 80.0}}

    calc = create_calculator(config)
    espresso_calc = cast(Espresso, calc)
    assert espresso_calc.parameters["input_data"]["system"]["ecutwfc"] == 80.0
    # Other defaults should remain
    assert espresso_calc.parameters["input_data"]["system"]["ecutrho"] == 200.0


def test_create_calculator_unsupported_code(config: DFTConfig) -> None:
    """Test error on unsupported DFT code."""
    # We cheat to bypass validation for this test
    # (Since we validate code in config, but calculator also checks)
    object.__setattr__(config, "code", "vasp")

    with pytest.raises(NotImplementedError, match="not supported"):
        create_calculator(config)


def test_create_calculator_invalid_parameters(config: DFTConfig) -> None:
    """Test error on invalid/restricted parameters."""
    config.parameters = {"INVALID_SECTION": {"key": "value"}}

    with pytest.raises(ValueError, match="Security Error"):
        create_calculator(config)
