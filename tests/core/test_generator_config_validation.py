"""Tests for StructureGeneratorConfig validation."""

import pytest
from pydantic import ValidationError

from pyacemaker.core.config import StructureGeneratorConfig


def test_valid_generator_config() -> None:
    """Test valid configuration values."""
    config = StructureGeneratorConfig(
        strain_range=0.15, rattle_amplitude=0.1, defect_density=0.01, initial_exploration="m3gnet"
    )
    assert config.strain_range == 0.15
    assert config.rattle_amplitude == 0.1
    assert config.defect_density == 0.01


def test_invalid_strain_range() -> None:
    """Test negative strain range raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        StructureGeneratorConfig(strain_range=-0.1)

    assert "Input should be greater than or equal to 0" in str(excinfo.value)


def test_invalid_rattle_amplitude() -> None:
    """Test negative rattle amplitude raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        StructureGeneratorConfig(rattle_amplitude=-0.5)

    assert "Input should be greater than or equal to 0" in str(excinfo.value)


def test_invalid_defect_density() -> None:
    """Test negative defect density raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        StructureGeneratorConfig(defect_density=-0.01)

    assert "Input should be greater than or equal to 0" in str(excinfo.value)


def test_invalid_initial_exploration_type() -> None:
    """Test invalid initial exploration type (Pydantic type checking)."""
    # Verify valid string
    config = StructureGeneratorConfig(initial_exploration="random")
    assert config.initial_exploration == "random"

    # Verify invalid type (int instead of str)
    # Pydantic attempts coercion for basic types, but let's check for obviously wrong types
    # or ensure we are testing the type constraint
    # Note: Pydantic V2 often coerces int to str "123", so checking None or list is better
    with pytest.raises(ValidationError):
        StructureGeneratorConfig(initial_exploration=None)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        StructureGeneratorConfig(initial_exploration=[])  # type: ignore[arg-type]
