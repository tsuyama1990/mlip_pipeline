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
        "simulation_goal": {"type": "melt_quench"},
    }
    user_config = UserInputConfig.model_validate(user_config_dict)

    system_config = ConfigFactory.from_user_input(user_config)

    assert system_config.project_name == "test_project"
    assert isinstance(system_config.run_uuid, UUID)
    assert system_config.explorer_config.fingerprint.species == ["Ni"]
    assert system_config.dft_config.dft_input_params.magnetism is not None
