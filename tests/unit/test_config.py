from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.base_config import GlobalConfig


def test_valid_config(temp_config: Path) -> None:
    config = GlobalConfig.from_yaml(temp_config)
    assert config.project_name == "test_project"
    assert config.oracle.type == "mock"


def test_invalid_config_type(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    with config_path.open("w") as f:
        f.write("project_name: test\noracle:\n  type: invalid_type")

    with pytest.raises(ValidationError) as exc:
        GlobalConfig.from_yaml(config_path)
    assert "Input should be 'mock' or 'quantum_espresso'" in str(exc.value)


def test_missing_field(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    with config_path.open("w") as f:
        f.write("oracle:\n  type: mock")

    with pytest.raises(ValidationError):
        GlobalConfig.from_yaml(config_path)
