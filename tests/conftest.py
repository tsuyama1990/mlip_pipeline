from pathlib import Path

import pytest
import yaml


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    config_data = {
        "project_name": "test_project",
        "orchestrator": {"max_iterations": 2},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
        "structure_generator": {"type": "mock"},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path
