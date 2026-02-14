"""Tests for configuration management."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from pyacemaker.core.config import (
    DFTConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    load_config,
)
from pyacemaker.core.exceptions import ConfigurationError


def test_project_config_valid() -> None:
    """Test valid ProjectConfig."""
    # Using a dummy path that is safe
    safe_path = Path("test_dir")
    config = ProjectConfig(name="Test", root_dir=safe_path)
    assert config.name == "Test"
    # Expect absolute path
    assert config.root_dir == safe_path.resolve()


def test_project_config_invalid() -> None:
    """Test invalid ProjectConfig (missing fields)."""
    with pytest.raises(ValidationError):
        ProjectConfig(name="Test")  # type: ignore[call-arg]


def test_project_config_path_traversal() -> None:
    """Test path traversal validation for root_dir."""
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        ProjectConfig(name="Test", root_dir=Path("../test"))

    with pytest.raises(ValueError, match="Path traversal not allowed"):
        ProjectConfig(name="Test", root_dir=Path("foo/../bar"))


def test_dft_config_default() -> None:
    """Test DFTConfig required fields."""
    with pytest.raises(ValidationError):
        DFTConfig()  # type: ignore[call-arg]

    config = DFTConfig(code="vasp")
    assert config.code == "vasp"


def test_dft_config_parameters_validation() -> None:
    """Test DFTConfig parameters validation."""
    # Pydantic validation error expected for incorrect key type
    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(code="vasp", parameters={1: "invalid"})  # type: ignore[dict-item]

    # Check that error relates to key type
    assert "Input should be a valid string" in str(excinfo.value)


def test_full_config_valid() -> None:
    """Test valid full configuration."""
    data = {
        "version": "0.1.0",
        "project": {"name": "Test", "root_dir": "test_dir"},
        "oracle": {"dft": {"code": "vasp"}},
    }
    config = PYACEMAKERConfig(**data)  # type: ignore[arg-type]
    assert config.project.name == "Test"
    assert config.oracle.dft.code == "vasp"
    # Test default logging
    assert config.logging.level == "INFO"


def test_version_validation() -> None:
    """Test semantic version validation."""
    data = {
        "version": "invalid",
        "project": {"name": "Test", "root_dir": "."},
        "oracle": {"dft": {"code": "vasp"}},
    }
    # Pydantic regex pattern validation error
    with pytest.raises(ValidationError) as excinfo:
        PYACEMAKERConfig(**data)  # type: ignore[arg-type]

    assert "String should match pattern" in str(excinfo.value)


def test_load_config_valid(tmp_path: Path) -> None:
    """Test loading a valid YAML configuration file."""
    config_data = {
        "version": "0.1.0",
        "project": {"name": "TestProject", "root_dir": str(tmp_path)},
        "oracle": {"dft": {"code": "quantum_espresso"}},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.project.name == "TestProject"
    assert config.project.root_dir == tmp_path.resolve()
    assert config.oracle.dft.code == "quantum_espresso"
    assert config.logging.level == "INFO"


def test_load_config_missing_file() -> None:
    """Test loading a non-existent file."""
    with pytest.raises(ConfigurationError):
        load_config(Path("non_existent.yaml"))


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Test loading an invalid YAML file."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("project: name: Test", encoding="utf-8")  # Invalid YAML structure

    with pytest.raises(ConfigurationError):
        load_config(config_file)


def test_load_config_invalid_structure(tmp_path: Path) -> None:
    """Test loading a YAML that is a list, not a dictionary."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must contain a YAML dictionary"):
        load_config(config_file)


def test_load_config_yaml_error(tmp_path: Path) -> None:
    """Test loading a malformed YAML file."""
    config_file = tmp_path / "malformed.yaml"
    config_file.write_text("key: value: error", encoding="utf-8")  # Invalid YAML syntax

    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        load_config(config_file)


def test_load_config_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of OSError during file read."""
    config_file = tmp_path / "valid.yaml"
    config_file.touch()

    def mock_open(*args: Any, **kwargs: Any) -> Any:
        msg = "Simulated read error"
        raise OSError(msg)

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    # Need to bypass size check because we are not mocking stat properly for strict typing environments
    # but monkeypatching Path.stat is risky as seen before.
    # Instead, we rely on the fact that the touched file is empty (size 0), so size check passes naturally.
    # The real issue in previous fail was the mock lambda missing st_mode.
    # Since we don't mock stat here, it uses real stat which is fine.

    with pytest.raises(ConfigurationError, match="Error reading configuration"):
        load_config(config_file)


def test_load_config_file_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a file that exceeds the size limit."""
    config_file = tmp_path / "large.yaml"
    config_file.touch()

    # Mock stat to return a large size
    original_stat = Path.stat

    def mock_stat(self: Path, *, follow_symlinks: bool = True) -> Any:
        # If it's our file, return large size
        if self == config_file:
            real_stat = original_stat(self, follow_symlinks=follow_symlinks)

            class MockStat:
                st_size = 20 * 1024 * 1024  # 20 MB
                st_mode = real_stat.st_mode

            return MockStat()
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("pathlib.Path.stat", mock_stat)

    with pytest.raises(ConfigurationError, match="Configuration file too large"):
        load_config(config_file)


def test_load_config_empty_file(tmp_path: Path) -> None:
    """Test loading an empty configuration file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must contain a YAML dictionary"):
        load_config(config_file)


def test_load_config_whitespace_file(tmp_path: Path) -> None:
    """Test loading a configuration file with only whitespace."""
    config_file = tmp_path / "whitespace.yaml"
    config_file.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must contain a YAML dictionary"):
        load_config(config_file)


def test_load_config_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of PermissionError during file read."""
    config_file = tmp_path / "protected.yaml"
    config_file.touch()

    def mock_open(*args: Any, **kwargs: Any) -> Any:
        msg = "Permission denied"
        raise PermissionError(msg)

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    with pytest.raises(ConfigurationError, match="Error reading configuration file"):
        load_config(config_file)


def test_load_config_is_directory(tmp_path: Path) -> None:
    """Test loading a path that is a directory, not a file."""
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()

    with pytest.raises(ConfigurationError, match="Configuration path is not a file"):
        load_config(config_dir)
