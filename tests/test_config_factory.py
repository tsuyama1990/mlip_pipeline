import pytest
import yaml
from pathlib import Path
from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.exceptions import ConfigError

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

    # Run in tmp_path context so Path.cwd() works as expected for resolving relative paths
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        system_config = ConfigFactory.from_yaml(input_file)

        assert isinstance(system_config, SystemConfig)
        assert system_config.minimal.project_name == "TestFactory"
        assert system_config.working_dir == tmp_path / "TestFactory"

        # KEY ASSERTION: Ensure no directories are created by the factory
        assert not system_config.working_dir.exists(), "ConfigFactory should not create directories!"

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

    with pytest.raises(ConfigError):
        ConfigFactory.from_yaml(input_file)

def test_file_not_found():
    with pytest.raises(ConfigError) as exc:
        ConfigFactory.from_yaml(Path("non_existent.yaml"))
    assert "Configuration file not found" in str(exc.value)

def test_invalid_yaml(tmp_path):
    input_file = tmp_path / "invalid.yaml"
    input_file.write_text("invalid: yaml: content: [")

    with pytest.raises(ConfigError) as exc:
        ConfigFactory.from_yaml(input_file)
    assert "Invalid YAML format" in str(exc.value)
