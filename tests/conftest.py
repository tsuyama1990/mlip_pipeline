from pathlib import Path

import pytest


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary directory for the project.
    """
    project_dir = tmp_path / "project_test"
    project_dir.mkdir()
    return project_dir
    # cleanup is handled by tmp_path, but strictly we could clean up here

@pytest.fixture
def dummy_dataset(temp_project_dir: Path) -> Path:
    """
    Creates a dummy dataset file.
    """
    data_path = temp_project_dir / "data.pckl"
    data_path.touch()
    return data_path

@pytest.fixture
def valid_config_yaml(temp_project_dir: Path, dummy_dataset: Path) -> Path:
    """
    Creates a valid config.yaml.
    """
    config_path = temp_project_dir / "config.yaml"
    config_content = f"""
project:
  name: "TestProject"
  seed: 123

training:
  dataset_path: "{dummy_dataset}"
  max_epochs: 10
  command: "echo"

orchestrator:
  max_iterations: 1
"""
    config_path.write_text(config_content)
    return config_path
