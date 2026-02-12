import os
from pathlib import Path

import pytest
import yaml

from mlip_autopipec.core.config_parser import ConfigError, load_config
from mlip_autopipec.domain_models.config import GlobalConfig


def test_load_valid_config(tmp_path: Path) -> None:
    config_data = {
        "orchestrator": {},
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert isinstance(config, GlobalConfig)


def test_load_config_env_var(tmp_path: Path) -> None:
    os.environ["TEST_VAR"] = "mock"
    # Create YAML manually to ensure ${TEST_VAR} is preserved as string
    yaml_content = """
    orchestrator: {}
    generator:
      type: ${TEST_VAR}
    oracle:
      type: mock
    trainer:
      type: mock
    dynamics:
      type: mock
    validator:
      type: mock
    """
    config_file = tmp_path / "config_env.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)
    assert config.generator.type == "mock"
    del os.environ["TEST_VAR"]


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_invalid_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: [ unclosed")
    with pytest.raises(ConfigError, match="Error loading config"):
        load_config(config_file)


def test_load_invalid_model(tmp_path: Path) -> None:
    config_data = {"orchestrator": {"max_cycles": -1}} # Invalid
    config_file = tmp_path / "invalid_model.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ConfigError, match="validation failed"):
        load_config(config_file)


def test_load_large_config(tmp_path: Path) -> None:
    config_file = tmp_path / "large_config.yaml"
    # Write > 1MB file
    size = 1024 * 1024 + 100
    with config_file.open("w") as f:
        f.write("a" * size)

    with pytest.raises(ConfigError, match="too large"):
        load_config(config_file)
