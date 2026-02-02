import pytest
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import Config


def test_load_valid_config(valid_config_yaml):
    """Test loading a valid configuration."""
    with valid_config_yaml.open("r") as f:
        data = yaml.safe_load(f)

    config = Config(**data)
    assert config.project.name == "TestProject"
    assert config.training.max_epochs == 10
    assert config.orchestrator.max_iterations == 1

def test_missing_dataset(temp_project_dir):
    """Test that missing dataset raises ValidationError."""
    config_path = temp_project_dir / "bad_config.yaml"
    config_content = """
project:
  name: "BadProject"
training:
  dataset_path: "ghost.pckl"
  command: "echo"
"""
    config_path.write_text(config_content)

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    # Dataset path validation happens at instantiation
    # Since ghost.pckl doesn't exist relative to CWD or absolute, it should fail.
    # We might need to handle CWD carefully. Pydantic FilePath checks existence.

    with pytest.raises(ValidationError) as excinfo:
        Config(**data)

    assert "path" in str(excinfo.value)

def test_invalid_epochs(temp_project_dir, dummy_dataset):
    """Test validation of numeric constraints."""
    config_data = {
        "project": {"name": "Test"},
        "training": {
            "dataset_path": str(dummy_dataset),
            "max_epochs": -5,
            "command": "echo"
        }
    }

    with pytest.raises(ValidationError):
        Config(**config_data)
