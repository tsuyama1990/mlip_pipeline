from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import pyacemaker.core.config  # Import the module to patch
from pyacemaker.core.config import (
    DynamicsEngineConfig,
    MaceConfig,
    get_defaults,
)
from pyacemaker.core.config_loader import load_config
from pyacemaker.core.exceptions import ConfigurationError


def test_load_valid_config(tmp_path):
    """Test loading a valid configuration file."""
    config_data = {
        "version": "0.1.0",
        "project": {"name": "test_project", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {"pseudopotentials": {"Fe": "Fe.upf"}},
            "mock": True
        },
        "trainer": {"potential_type": "pace"},
    }

    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.version == "0.1.0"
    assert config.project.name == "test_project"

def test_load_missing_config():
    """Test error when config file missing."""
    with pytest.raises(ConfigurationError):
        load_config(Path("nonexistent.yaml"))

def test_defaults_loading(tmp_path, monkeypatch):
    """Test loading defaults from yaml."""
    defaults_data = {
        "dataset": {"train_ratio": 0.9}
    }
    defaults_file = tmp_path / "defaults.yaml"
    with defaults_file.open("w") as f:
        yaml.dump(defaults_data, f)

    # Correctly patch the module-level variable
    monkeypatch.setattr(pyacemaker.core.config, "_DEFAULTS_PATH", defaults_file)

    # Clear lru_cache
    get_defaults.cache_clear()

    try:
        defaults = get_defaults()
        assert defaults["dataset"]["train_ratio"] == 0.9
    finally:
        # Restore cache state for other tests
        get_defaults.cache_clear()

def test_mace_config_url_validation():
    """Test URL validation in MaceConfig."""
    # Valid URL
    cfg = MaceConfig(model_path="https://github.com/mace-model/mace/releases/download/v0.1/model.model")
    assert cfg.model_path.startswith("https://")

    # Invalid URL (spaces)
    with pytest.raises(ValidationError):
        MaceConfig(model_path="https://example.com/bad url")

    # Invalid URL (no scheme) - treated as file path, so should pass validation
    # when skip_file_checks is True (default in tests)
    cfg = MaceConfig(model_path="example.com/model")
    assert str(cfg.model_path).endswith("example.com/model")

def test_dynamics_config_sanitization():
    """Test parameter sanitization in DynamicsEngineConfig."""
    # Safe params
    cfg = DynamicsEngineConfig(parameters={"thermo": 100})
    assert cfg.parameters["thermo"] == 100

    # Unsafe params (injection attempt)
    with pytest.raises(ValidationError):
        DynamicsEngineConfig(parameters={"dump": "100; rm -rf /"})

    with pytest.raises(ValidationError):
         DynamicsEngineConfig(parameters={"fix": "1 && echo hacked"})

