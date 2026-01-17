
import pytest
from pydantic import ValidationError

from mlip_autopipec.config.models import UserInputConfig


def test_valid_user_config_parses_correctly():
    """Tests that a valid user configuration is parsed without errors."""
    valid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {
            "type": "melt_quench",
            "temperature_range": [300, 2000],
        },
    }
    try:
        UserInputConfig.model_validate(valid_config)
    except ValidationError as e:
        pytest.fail(f"Valid configuration failed to parse: {e}")


def test_composition_sum_not_one_raises_error():
    """Tests that a validation error is raised if composition fractions do not sum to 1.0."""
    invalid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.4},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
    }
    # With RootModel, the error location might be slightly different in message, but validation happens
    with pytest.raises(ValidationError, match="Composition fractions must sum to 1.0"):
        UserInputConfig.model_validate(invalid_config)


def test_composition_elements_mismatch_raises_error():
    """Tests that a validation error is raised if composition keys do not match elements."""
    invalid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Cr": 0.5},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
    }
    with pytest.raises(ValidationError, match="Composition keys must match the elements list"):
        UserInputConfig.model_validate(invalid_config)


def test_invalid_element_symbol_raises_error():
    """Tests that a validation error is raised for an invalid chemical symbol."""
    invalid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe", "Xy"],
            "composition": {"Fe": 0.5, "Xy": 0.5},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
    }
    with pytest.raises(ValidationError, match="'Xy' is not a valid chemical symbol"):
        UserInputConfig.model_validate(invalid_config)


def test_extra_field_raises_error():
    """Tests that an error is raised for an unexpected field, enforcing the 'forbid' extra config."""
    invalid_config = {
        "project_name": "Test Project",
        "project_version": "1.0",  # Extra field
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        UserInputConfig.model_validate(invalid_config)
