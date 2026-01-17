from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import SystemConfig


def test_config_factory_valid_yaml(tmp_path: Path) -> None:
    config_data = {
        "project_name": "TestFactory",
        "target_system": {
            "elements": ["Au"],
            "composition": {"Au": 1.0}
        },
        "resources": {
            "dft_code": "vasp",
            "parallel_cores": 4
        }
    }
    input_file = tmp_path / "input.yaml"
    with input_file.open("w") as f:
        yaml.dump(config_data, f)

    config = ConfigFactory.from_yaml(input_file)

    assert isinstance(config, SystemConfig)
    assert config.minimal.project_name == "TestFactory"
    assert config.working_dir.exists()
    assert config.working_dir.name == "TestFactory"
    assert config.db_path.name == "TestFactory.db"
    assert config.log_path.name == "system.log"
    assert config.db_path.parent == config.working_dir
    assert config.log_path.parent == config.working_dir


def test_config_factory_invalid_yaml(tmp_path: Path) -> None:
    config_data = {
        "project_name": "BadConfig",
        "target_system": {
            "elements": ["Au"],
            "composition": {"Au": 0.5} # Invalid sum
        },
        "resources": {
            "dft_code": "vasp",
            "parallel_cores": 4
        }
    }
    input_file = tmp_path / "bad_input.yaml"
    with input_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        ConfigFactory.from_yaml(input_file)


def test_config_factory_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        ConfigFactory.from_yaml(Path("nonexistent.yaml"))
