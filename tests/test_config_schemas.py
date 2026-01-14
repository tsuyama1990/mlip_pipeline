"""Unit tests for the Pydantic configuration schemas."""

import pytest
from pydantic import ValidationError

from mlip_autopipec.config_schemas import (
    DFTConfig,
    Resources,
    SystemConfig,
    TargetSystem,
    UserConfig,
)


def test_target_system_valid() -> None:
    """Test that a valid TargetSystem model is parsed correctly."""
    data = {"elements": ["Ni"], "composition": {"Ni": 1.0}}
    system = TargetSystem(**data)
    assert system.elements == ["Ni"]
    assert system.composition == {"Ni": 1.0}


def test_target_system_invalid_composition_sum() -> None:
    """Test that a composition not summing to 1.0 raises a validation error."""
    data = {"elements": ["Ni", "Fe"], "composition": {"Ni": 0.5, "Fe": 0.6}}
    with pytest.raises(ValidationError, match="Composition fractions must sum to 1.0"):
        TargetSystem(**data)


def test_target_system_mismatched_elements() -> None:
    """Test that mismatched elements and composition keys raise a validation error."""
    data = {"elements": ["Ni", "Fe"], "composition": {"Ni": 0.5, "Al": 0.5}}
    with pytest.raises(
        ValidationError, match="Elements in composition must match the elements list"
    ):
        TargetSystem(**data)


def test_user_config_valid_defaults() -> None:
    """Test a valid UserConfig and ensure Resources gets its default value."""
    data = {
        "target_system": {"elements": ["Si"], "composition": {"Si": 1.0}},
        "simulation_goal": "melt_quench",
    }
    config = UserConfig(**data)
    assert config.simulation_goal == "melt_quench"
    assert isinstance(config.resources, Resources)
    assert config.resources.dft_cores == 1


def test_user_config_extra_field_forbidden() -> None:
    """Test that an extra, undefined field raises a validation error."""
    data = {
        "target_system": {"elements": ["Si"], "composition": {"Si": 1.0}},
        "simulation_goal": "elastic",
        "extra_parameter": "should_fail",
    }
    with pytest.raises(ValidationError) as exc_info:
        UserConfig(**data)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_pseudopotentials_invalid_element() -> None:
    """Test that an invalid chemical symbol in pseudopotentials raises an error."""
    dft_input = {
        "pseudopotentials": {"Xx": "Xx.UPF"},
    }
    with pytest.raises(ValidationError, match="'Xx' is not a valid chemical symbol"):
        DFTConfig(input=dft_input)


def test_system_config_valid() -> None:
    """Test a valid SystemConfig instantiation."""
    dft_config = {
        "input": {"pseudopotentials": {"Si": "Si.UPF"}},
    }
    config = SystemConfig(dft=dft_config, db_path="test.db")
    assert config.db_path == "test.db"
    assert isinstance(config.dft, DFTConfig)
    assert config.dft.executable.command == "pw.x"


def test_system_config_extra_field_forbidden() -> None:
    """Test that an extra, undefined field in SystemConfig raises an error."""
    config_data = {
        "dft": {"input": {"pseudopotentials": {"Si": "Si.UPF"}}},
        "db_path": "test.db",
        "extra_param": "should_fail",
    }
    with pytest.raises(ValidationError) as exc_info:
        SystemConfig(**config_data)
    assert "Extra inputs are not permitted" in str(exc_info.value)
