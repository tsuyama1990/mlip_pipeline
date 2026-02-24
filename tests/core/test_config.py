"""Tests for configuration loading and validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig, ProjectConfig
from pyacemaker.core.config_loader import load_config
from pyacemaker.core.exceptions import ConfigurationError


def test_load_valid_config(tmp_path: Path) -> None:
    """Test loading a valid configuration file."""
    config_data = {
        "version": "0.1.0",
        "project": {"name": "Test Project", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "quantum_espresso",
                "command": "pw.x",
                "pseudopotentials": {"Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"},
            },
            "mace": {"model_path": "medium"},
        },
        "logging": {"level": "DEBUG"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Create dummy pseudopotential file to pass file checks
    (tmp_path / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF").touch()

    config = load_config(config_file)
    assert isinstance(config, PYACEMAKERConfig)
    assert config.project.name == "Test Project"
    assert config.oracle.dft.code == "quantum_espresso"


def test_project_config_path_traversal() -> None:
    """Test path traversal check."""
    # Direct
    with pytest.raises(ValidationError, match="Path traversal detected"):
        ProjectConfig(name="test", root_dir=Path("../../../etc/passwd"))

    # Indirect
    with pytest.raises(ValidationError, match="Path traversal detected"):
        ProjectConfig(name="test", root_dir=Path("safe/../../unsafe"))


def test_missing_config_file(tmp_path: Path) -> None:
    """Test handling of missing configuration file."""
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    """Test handling of invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("key: value: invalid")
    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        load_config(config_file)


def test_invalid_schema(tmp_path: Path) -> None:
    """Test handling of schema validation errors."""
    config_data = {
        "version": "0.1.0",
        # Missing project
        "oracle": {},
    }
    config_file = tmp_path / "invalid_schema.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(config_file)


def test_load_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading defaults from environment variable path."""
    defaults = {"version": "0.1.0", "log_level": "INFO"}
    defaults_file = tmp_path / "defaults.yaml"
    with defaults_file.open("w") as f:
        yaml.dump(defaults, f)

    from pyacemaker.core.config import get_defaults
    get_defaults.cache_clear()

    # Patch _DEFAULTS_PATH directly because it's evaluated at import time
    with patch("pyacemaker.core.config._DEFAULTS_PATH", defaults_file):
        data = get_defaults()
        assert data["version"] == "0.1.0"


def test_defaults_file_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that too large defaults file raises error."""
    defaults_file = tmp_path / "large_defaults.yaml"
    # Create file slightly larger than limit (1MB)
    with defaults_file.open("wb") as f:
        f.write(b" " * (1024 * 1024 + 1))

    from pyacemaker.core.config import get_defaults
    get_defaults.cache_clear()

    with patch("pyacemaker.core.config._DEFAULTS_PATH", defaults_file):
        with pytest.raises(ValueError, match="Defaults file size"):
            get_defaults()


def test_defaults_not_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test defaults file not being a dictionary."""
    defaults_file = tmp_path / "list_defaults.yaml"
    defaults_file.write_text("- item1\n- item2")

    from pyacemaker.core.config import get_defaults
    get_defaults.cache_clear()

    with patch("pyacemaker.core.config._DEFAULTS_PATH", defaults_file):
        with pytest.raises(TypeError, match="must contain a YAML dictionary"):
            get_defaults()


def test_extra_fields_forbidden(tmp_path: Path) -> None:
    """Test that extra fields are forbidden."""
    original_skip = CONSTANTS.skip_file_checks
    CONSTANTS.skip_file_checks = True
    try:
        config_data = {
            "version": "0.1.0",
            "project": {"name": "Test", "root_dir": ".", "extra_field": "forbidden"},
            "oracle": {
                "dft": {"code": "vasp", "pseudopotentials": {"Fe": "Fe.pbe.UPF"}},
                "mock": True,
            },
        }
        config_file = tmp_path / "extra.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
            load_config(config_file)
    finally:
        CONSTANTS.skip_file_checks = original_skip


def test_load_config_permission_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of permission error."""
    # We must enable checks to test os.access check
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", False)
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "protected.yaml"
    config_file.touch()

    # Use relative path
    rel_path = Path("protected.yaml")

    # Mock os.access to return False
    monkeypatch.setattr(os, "access", lambda path, mode: False)

    with pytest.raises(ConfigurationError, match="Permission denied"):
        load_config(rel_path)
