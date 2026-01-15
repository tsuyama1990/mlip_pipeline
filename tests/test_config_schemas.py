"""Unit tests for the Pydantic configuration schemas."""

from typing import Any

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


def test_inference_params_valid() -> None:
    """Test valid InferenceParams configuration."""
    from mlip_autopipec.config_schemas import InferenceParams

    params = InferenceParams(
        md_ensemble={"ensemble_type": "npt", "target_temperature_k": 500.0},
        uncertainty_threshold=5.0,
        simulation_timestep_fs=0.5,
        total_simulation_steps=5000,
    )
    assert params.md_ensemble.ensemble_type == "npt"
    assert params.total_simulation_steps == 5000


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"total_simulation_steps": 0},
        {"simulation_timestep_fs": 0.0},
        {"uncertainty_threshold": -1.0},
        {"md_ensemble": {"target_temperature_k": -100.0}},
    ],
)
def test_inference_params_invalid(invalid_data: dict[str, Any]) -> None:
    """Test invalid InferenceParams that should raise ValidationError."""
    from mlip_autopipec.config_schemas import InferenceParams

    valid_data = {
        "md_ensemble": {"ensemble_type": "nvt", "target_temperature_k": 300.0},
        "uncertainty_threshold": 4.0,
        "simulation_timestep_fs": 1.0,
        "total_simulation_steps": 1000,
    }
    for key, value in invalid_data.items():
        if isinstance(value, dict) and isinstance(valid_data.get(key), dict):
            valid_data[key].update(value)  # type: ignore
        else:
            valid_data[key] = value

    with pytest.raises(ValidationError):
        InferenceParams(**valid_data)


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
    target_system = {"elements": ["Si"], "composition": {"Si": 1.0}}
    dft_config = {
        "input": {"pseudopotentials": {"Si": "Si.UPF"}},
    }
    config = SystemConfig(target_system=target_system, dft=dft_config, db_path="test.db")
    assert config.db_path == "test.db"
    assert isinstance(config.dft, DFTConfig)
    assert config.dft.executable.command == "pw.x"


def test_system_config_extra_field_forbidden() -> None:
    """Test that an extra, undefined field in SystemConfig raises an error."""
    config_data = {
        "target_system": {"elements": ["Si"], "composition": {"Si": 1.0}},
        "dft": {"input": {"pseudopotentials": {"Si": "Si.UPF"}}},
        "db_path": "test.db",
        "extra_param": "should_fail",
    }
    with pytest.raises(ValidationError) as exc_info:
        SystemConfig(**config_data)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_fps_params_invalid_num_structures() -> None:
    """Test that num_structures_to_select < 1 raises a validation error."""
    from mlip_autopipec.config_schemas import FPSParams

    with pytest.raises(ValidationError) as exc_info:
        FPSParams(num_structures_to_select=0)
    assert "Input should be greater than or equal to 1" in str(exc_info.value)


def test_explorer_params_missing_surrogate_model() -> None:
    """Test that ExplorerParams raises a validation error.

    This happens if surrogate_model is missing.
    """
    from mlip_autopipec.config_schemas import ExplorerParams

    with pytest.raises(ValidationError) as exc_info:
        ExplorerParams()  # type: ignore
    assert "Field required" in str(exc_info.value)


def test_surrogate_model_path_validation() -> None:
    """Test the path traversal validation for the surrogate model path."""
    from mlip_autopipec.config_schemas import SurrogateModelParams

    with pytest.raises(ValidationError, match="cannot contain '..'"):
        SurrogateModelParams(model_path="../path/to/model")

    with pytest.raises(ValidationError, match="must be a relative path"):
        SurrogateModelParams(model_path="/path/to/model")

    # A valid relative path should pass
    valid_path = "models/mace.model"
    params = SurrogateModelParams(model_path=valid_path)
    assert params.model_path == valid_path


def test_generator_params_valid() -> None:
    """Test that valid GeneratorParams are parsed correctly."""
    from mlip_autopipec.config_schemas import GeneratorParams

    params = GeneratorParams(
        alloy_params={"sqs_supercell_size": [2, 2, 2]},
        crystal_params={"defect_types": ["vacancy", "interstitial"]},
    )
    assert params.alloy_params.sqs_supercell_size == [2, 2, 2]
    assert "interstitial" in params.crystal_params.defect_types


def test_trainer_params_valid() -> None:
    """Test that valid TrainerParams are parsed correctly."""
    from mlip_autopipec.config_schemas import TrainerParams

    params = TrainerParams(
        loss_weights={"energy": 10.0, "forces": 1.0, "stress": 0.1},
        ace_params={"correlation_order": 4},
    )
    assert params.loss_weights.energy == 10.0
    assert params.ace_params.correlation_order == 4


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"loss_weights": {"energy": 0}},
        {"ace_params": {"correlation_order": 1}},
    ],
)
def test_trainer_params_invalid(invalid_data: dict[str, Any]) -> None:
    """Test that invalid TrainerParams raise a ValidationError."""
    from mlip_autopipec.config_schemas import TrainerParams

    valid_data = {
        "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 10.0},
        "ace_params": {"correlation_order": 3},
    }
    for key, value in invalid_data.items():
        if isinstance(value, dict) and isinstance(valid_data.get(key), dict):
            valid_data[key].update(value)  # type: ignore
        else:
            valid_data[key] = value

    with pytest.raises(ValidationError):
        TrainerParams(**valid_data)
