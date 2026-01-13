"""Unit tests for the Pydantic schemas."""

import pytest
from pydantic import ValidationError

from mlip_autopipec.schemas.user_config import GenerationConfig, TargetSystem, UserConfig


def test_user_config_valid() -> None:
    """Tests successful validation of UserConfig with correct data."""
    valid_data = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
    }
    config = UserConfig(**valid_data)
    assert config.project_name == "TestProject"
    assert config.target_system.crystal_structure == "fcc"


def test_user_config_mismatched_elements() -> None:
    """Tests for ValidationError when elements and composition keys do not match."""
    invalid_data = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Fe", "Cr"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
    }
    with pytest.raises(ValidationError, match="`elements` and `composition` keys do not match."):
        UserConfig(**invalid_data)


def test_user_config_composition_sum_invalid() -> None:
    """Tests for ValidationError when composition fractions do not sum to 1.0."""
    invalid_data = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.6},
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
    }
    with pytest.raises(ValidationError, match="Composition fractions must sum to 1.0"):
        UserConfig(**invalid_data)


def test_target_system_extra_fields_forbidden() -> None:
    """Tests that extra fields are forbidden in TargetSystem."""
    with pytest.raises(ValidationError):
        TargetSystem(
            elements=["Si"],
            composition={"Si": 1.0},
            crystal_structure="diamond",
            extra_field="should_fail",  # type: ignore[call-arg]
        )


def test_generation_config_extra_fields_forbidden() -> None:
    """Tests that extra fields are forbidden in GenerationConfig."""
    with pytest.raises(ValidationError):
        GenerationConfig(
            generation_type="eos",
            another_field="should_also_fail",  # type: ignore[call-arg]
        )
