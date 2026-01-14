"""Unit tests for the Pydantic configuration schemas."""

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.user import Resources, TargetSystem, UserConfig


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
