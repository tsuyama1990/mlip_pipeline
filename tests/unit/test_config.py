from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.loader import load_config


def test_load_valid_config(temp_dir: Path) -> None:
    data_file = temp_dir / "data.pckl"
    data_file.touch()

    config_data = {
        "project": {"name": "TestProject", "seed": 123},
        "training": {"dataset_path": str(data_file), "max_epochs": 50, "command": "mock_train"},
        "orchestrator": {"max_iterations": 5},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    assert config.project.name == "TestProject"
    assert config.project.seed == 123
    assert config.training.dataset_path == data_file
    assert config.training.max_epochs == 50
    assert config.orchestrator.max_iterations == 5


def test_load_config_missing_file(temp_dir: Path) -> None:
    config_data = {
        "project": {"name": "TestProject"},
        "training": {
            "dataset_path": str(temp_dir / "missing.pckl"),
        },
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)

    assert "dataset_path" in str(excinfo.value)


def test_load_config_invalid_yaml(temp_dir: Path) -> None:
    config_file = temp_dir / "bad.yaml"
    config_file.write_text("project: : : invalid", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(config_file)
