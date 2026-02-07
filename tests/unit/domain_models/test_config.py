from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    MockDynamicsConfig,
    MockOracleConfig,
    MockTrainerConfig,
)


def test_config_valid_mock(tmp_path: Path) -> None:
    """Test loading a valid configuration with mock components."""
    config_content = """
    workdir: "experiments/test_01"
    oracle:
      type: "mock"
      noise_level: 0.1
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    """
    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    config = GlobalConfig.from_yaml(p)
    assert isinstance(config.oracle, MockOracleConfig)
    assert config.oracle.noise_level == 0.1
    assert isinstance(config.trainer, MockTrainerConfig)
    assert isinstance(config.dynamics, MockDynamicsConfig)
    assert config.workdir == Path("experiments/test_01")


def test_config_invalid_type(tmp_path: Path) -> None:
    """Test validation error for invalid component type."""
    config_content = """
    workdir: "experiments/test_01"
    oracle:
      type: "unknown"
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    """
    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    with pytest.raises(ValidationError):
        GlobalConfig.from_yaml(p)


def test_config_file_not_found(tmp_path: Path) -> None:
    """Test error for missing configuration file."""
    p = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        GlobalConfig.from_yaml(p)


def test_config_missing_field(tmp_path: Path) -> None:
    """Test validation error for missing required field."""
    config_content = """
    workdir: "experiments/test_01"
    oracle:
      type: "mock"
    # trainer missing
    dynamics:
      type: "mock"
    """
    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    with pytest.raises(ValidationError):
        GlobalConfig.from_yaml(p)
