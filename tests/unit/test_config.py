from pathlib import Path

import pytest
import yaml

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
    assert config.training.max_epochs == 50
    assert config.orchestrator.max_iterations == 5
    # Default factories
    assert config.exploration.strategy == "random"
    assert config.oracle.method == "dft"
    assert config.validation.run_validation is True
    assert config.dft is None


def test_load_missing_file(temp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(temp_dir / "missing.yaml")


def test_load_invalid_yaml(temp_dir: Path) -> None:
    config_file = temp_dir / "bad.yaml"
    with config_file.open("w") as f:
        f.write("project: : : invalid")

    with pytest.raises(yaml.YAMLError):
        load_config(config_file)
