import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.schemas.data import StructureRecord
from mlip_autopipec.schemas.user_config import (
    GenerationConfig,
    TargetSystem,
    UserConfig,
)


def test_user_config_valid() -> None:
    """Test that a valid UserConfig model can be created."""
    valid_data = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
        "surrogate_config": {
            "model_path": "path/to/model",
            "num_to_select_fps": 10,
            "descriptor_type": "SOAP",
        },
        "trainer_config": {
            "radial_basis": "bessel",
            "max_body_order": 2,
            "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 0.0},
        },
    }
    config = UserConfig.model_validate(valid_data)
    assert config.project_name == "test_project"
    assert config.target_system.crystal_structure == "fcc"


def test_user_config_composition_sum_invalid() -> None:
    """Test that a composition not summing to 1.0 raises a ValueError."""
    invalid_data = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.6},  # Sums to 1.1
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
        "surrogate_config": {
            "model_path": "path/to/model",
            "num_to_select_fps": 10,
            "descriptor_type": "SOAP",
        },
        "trainer_config": {
            "radial_basis": "bessel",
            "max_body_order": 2,
            "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 0.0},
        },
    }
    with pytest.raises(ValidationError, match="composition values must sum to 1.0"):
        UserConfig.model_validate(invalid_data)


def test_user_config_elements_composition_mismatch() -> None:
    """Test that a mismatch between elements and composition keys raises a ValueError."""
    invalid_data = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Co": 0.5},  # Co instead of Ni
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
        "surrogate_config": {
            "model_path": "path/to/model",
            "num_to_select_fps": 10,
            "descriptor_type": "SOAP",
        },
        "trainer_config": {
            "radial_basis": "bessel",
            "max_body_order": 2,
            "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 0.0},
        },
    }
    with pytest.raises(ValidationError, match="elements and composition keys must match"):
        UserConfig.model_validate(invalid_data)


def test_target_system_extra_fields_forbidden() -> None:
    """Test that extra fields are not allowed in TargetSystem."""
    with pytest.raises(ValidationError):
        TargetSystem(
            elements=["Si"],
            composition={"Si": 1.0},
            crystal_structure="diamond",
            extra_field="should_fail",  # type: ignore
        )


def test_generation_config_extra_fields_forbidden() -> None:
    """Test that extra fields are not allowed in GenerationConfig."""
    with pytest.raises(ValidationError):
        GenerationConfig(
            generation_type="eos",
            another_field="should_fail",  # type: ignore
        )


def test_structure_record_valid() -> None:
    """Test that a valid StructureRecord model can be created."""
    atoms = Atoms("H", positions=[(0, 0, 0)])
    record = StructureRecord(
        atoms=atoms,
        config_type="test",
        source="test",
        surrogate_energy=0.0,
    )
    assert record.config_type == "test"
