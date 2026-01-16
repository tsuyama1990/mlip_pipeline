"""Tests for the PacemakerConfigGenerator module."""

from pathlib import Path

import pytest
import yaml

from mlip_autopipec.config.models import PacemakerConfig, SystemConfig
from mlip_autopipec.modules.config_generator import PacemakerConfigGenerator


@pytest.fixture
def test_system_config(tmp_path: Path) -> SystemConfig:
    """Provide a default SystemConfig for testing."""
    user_config_dict = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Ni"],
            "composition": {"Ni": 1.0},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
    }
    from mlip_autopipec.config.factory import ConfigFactory
    from mlip_autopipec.config.models import UserInputConfig

    user_config = UserInputConfig.model_validate(user_config_dict)
    system_config = ConfigFactory.from_user_input(user_config)

    # We need to create some dummy files for the paths
    (tmp_path / "pacemaker").touch()
    (tmp_path / "lammps").touch()
    (tmp_path / "template.in").touch()
    (tmp_path / "potential.yace").touch()

    system_config.training_config.pacemaker_executable = tmp_path / "pacemaker"
    system_config.training_config.template_file = tmp_path / "template.in"
    system_config.inference_config.lammps_executable = tmp_path / "lammps"
    system_config.inference_config.potential_path = tmp_path / "potential.yace"

    return system_config


def test_generate_pacemaker_config(test_system_config: SystemConfig, tmp_path: Path) -> None:
    """Unit test for the Pacemaker config generation logic."""
    generator = PacemakerConfigGenerator(test_system_config)
    dummy_data_path = tmp_path / "dummy_data.xyz"
    dummy_data_path.touch()

    config_path = generator.generate_config(dummy_data_path, tmp_path)
    assert config_path.exists()

    # Verify that the generated YAML can be loaded and parsed by the Pydantic model
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
        parsed_config = PacemakerConfig(**config_data)

    # Verify that the generated config matches the SystemConfig
    fit_params = parsed_config.fit_params
    assert fit_params.dataset_filename == str(dummy_data_path)
    assert fit_params.loss_weights.energy == test_system_config.training_config.loss_weights.energy
    assert (
        fit_params.ace.correlation_order
        == test_system_config.training_config.ace_params.correlation_order
    )
