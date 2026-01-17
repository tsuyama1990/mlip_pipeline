"""Unit tests for the ConfigFactory module."""

from uuid import UUID

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import UserInputConfig

def test_config_factory_from_user_input():
    """Test that the ConfigFactory correctly expands a UserInputConfig."""
    user_config_dict = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Ni"],
            "composition": {"Ni": 1.0},
            "crystal_structure": "fcc",
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        },
        # simulation_goal is not in MinimalConfig anymore, ignoring it
    }
    user_config = UserInputConfig.model_validate(user_config_dict)

    system_config = ConfigFactory.from_user_input(user_config)

    assert system_config.minimal.project_name == "test_project"
    assert system_config.project_name == "test_project" # Optional field populated
    assert isinstance(system_config.run_uuid, UUID)

    # Check heuristics population
    # Note: explorer_config is optional in SystemConfig but factory populates it
    assert system_config.explorer_config is not None
    assert system_config.explorer_config.fingerprint.species == ["Ni"]

    assert system_config.dft_config is not None
    assert system_config.dft_config.dft_input_params.magnetism is not None
