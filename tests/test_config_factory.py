import pytest
import yaml

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.exceptions import ConfigError


def test_config_factory_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConfigFactory.from_yaml(tmp_path / "nonexistent.yaml")


def test_config_factory_invalid_yaml(tmp_path):
    p = tmp_path / "invalid.yaml"
    # Create definitely invalid YAML
    p.write_text("key: value\n\tinvalid tab")

    with pytest.raises(ConfigError) as excinfo:
        ConfigFactory.from_yaml(p)
    assert "Failed to parse YAML" in str(excinfo.value)


def test_config_factory_invalid_schema(tmp_path):
    p = tmp_path / "bad_schema.yaml"
    data = {"project_name": "Test"}  # Missing other required fields
    with p.open("w") as f:
        yaml.dump(data, f)

    with pytest.raises(ConfigError) as excinfo:
        ConfigFactory.from_yaml(p)
    assert "Configuration validation failed" in str(excinfo.value)
