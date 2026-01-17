import pytest
import yaml
from pathlib import Path
from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.schemas.system import SystemConfig
from pydantic import ValidationError

def test_from_yaml_valid(tmp_path):
    # Create a dummy input.yaml
    input_file = tmp_path / "input.yaml"
    config_data = {
        "project_name": "TestFactory",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        },
        "simulation_goal": {
            "type": "melt_quench"
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    # Change cwd to tmp_path so the factory creates the project dir there
    # But ConfigFactory uses Path.cwd(). We need to mock Path.cwd() or run inside a block.
    # Using monkeypatch for Path.cwd() is tricky because it's a method.
    # Instead, we can run the test such that the "current directory" is tmp_path.

    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        system_config = ConfigFactory.from_yaml(input_file)

        assert isinstance(system_config, SystemConfig)
        assert system_config.minimal.project_name == "TestFactory"
        assert system_config.working_dir == tmp_path / "TestFactory"
        assert system_config.db_path == tmp_path / "TestFactory" / "project.db"
        assert system_config.working_dir.exists()

    finally:
        os.chdir(original_cwd)

def test_from_yaml_invalid(tmp_path):
    input_file = tmp_path / "bad.yaml"
    config_data = {
        "project_name": "BadProject",
        "target_system": { # Missing composition
            "elements": ["Al"]
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        ConfigFactory.from_yaml(input_file)

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        ConfigFactory.from_yaml(Path("non_existent.yaml"))
