"""Unit tests for the ConfigFactory module."""

from uuid import UUID

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import UserInputConfig

RESOURCES_DEFAULT = {
    "dft_code": "quantum_espresso",
    "parallel_cores": 4,
    "gpu_enabled": False
}

def test_config_factory_creates_paths_and_config(tmp_path):
    """Test that the ConfigFactory correctly expands a UserInputConfig and sets up paths."""
    user_config_dict = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Ni"],
            "composition": {"Ni": 1.0},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
        "resources": RESOURCES_DEFAULT
    }
    user_config = UserInputConfig.model_validate(user_config_dict)

    # We pass base_dir to avoid polluting the actual filesystem
    system_config = ConfigFactory.from_user_input(user_config, base_dir=tmp_path)

    assert isinstance(system_config.run_uuid, UUID)
    assert system_config.user_input == user_config

    # Check paths
    expected_work_dir = tmp_path / "test_project"
    assert system_config.working_dir == expected_work_dir
    assert system_config.working_dir.exists()

    assert system_config.db_path == expected_work_dir / "test_project.db"
    assert system_config.log_path == expected_work_dir / "system.log"

    # Check absolute
    assert system_config.db_path.is_absolute()
    assert system_config.log_path.is_absolute()
