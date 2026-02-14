"""Tests for configuration management."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    load_config,
)
from pyacemaker.core.exceptions import ConfigurationError


def test_constants_defaults() -> None:
    """Test that constants have expected default values."""
    assert CONSTANTS.default_version == "0.1.0"
    assert CONSTANTS.max_config_size == 1 * 1024 * 1024


def test_project_config_valid() -> None:
    """Test valid ProjectConfig."""
    safe_path = Path("test_dir")
    config = ProjectConfig(name="Test", root_dir=safe_path)
    assert config.name == "Test"
    # Depending on environment, resolve might be absolute.
    # We check it resolves successfully.
    assert config.root_dir.is_absolute() or config.root_dir == safe_path


def test_project_config_path_traversal() -> None:
    """Test path traversal validation for root_dir."""
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        ProjectConfig(name="Test", root_dir=Path("../test"))


def test_dft_config_parameters_validation() -> None:
    """Test DFTConfig parameters validation."""
    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(code="vasp", parameters={1: "invalid"})  # type: ignore[dict-item]
    # Pydantic's default validation message for incorrect dict key type
    assert "Input should be a valid string" in str(excinfo.value)


def test_version_validation() -> None:
    """Test semantic version validation."""
    data = {
        "version": "invalid",
        "project": {"name": "Test", "root_dir": "."},
        "oracle": {"dft": {"code": "vasp"}},
    }
    with pytest.raises(ValidationError) as excinfo:
        PYACEMAKERConfig(**data)  # type: ignore[arg-type]
    # Pydantic's regex mismatch message
    assert "String should match pattern" in str(excinfo.value)


def test_load_config_file_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a file that exceeds the size limit."""
    config_file = tmp_path / "large.yaml"
    config_file.touch()

    # Mock stat to return a large size

    class MockStat:
        st_size = CONSTANTS.max_config_size + 1
        st_mode = 33188 # Regular file mode

    # Fix: Accept **kwargs to handle follow_symlinks argument passed by pathlib/pytest
    monkeypatch.setattr("pathlib.Path.stat", lambda self, **kwargs: MockStat())

    with pytest.raises(ConfigurationError, match="Configuration file too large"):
        load_config(config_file)


def test_load_config_content_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that reading stops if content exceeds limit (race condition simulation)."""
    config_file = tmp_path / "race_large.yaml"

    # Write content slightly larger than limit
    content = "a" * (CONSTANTS.max_config_size + 10)
    config_file.write_text(content)

    # We skip the stat check to simulate race condition where file grew
    class MockStat:
        st_size = 100 # Small size reported
        st_mode = 33188

    monkeypatch.setattr("pathlib.Path.stat", lambda self, **kwargs: MockStat())

    with pytest.raises(ConfigurationError, match="exceeds limit"):
        load_config(config_file)


def test_load_config_chunked_read(tmp_path: Path) -> None:
    """Test reading a valid file in chunks (implicit test of logic)."""
    config_data = {
        "version": "0.1.0",
        "project": {"name": "ChunkTest", "root_dir": str(tmp_path)},
        "oracle": {"dft": {"code": "vasp"}}
    }
    config_file = tmp_path / "chunk_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.project.name == "ChunkTest"


def test_load_config_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of OSError during file read."""
    config_file = tmp_path / "valid.yaml"
    config_file.touch()

    # We mock open to raise OSError.
    # We allow stat to pass (default behavior on touched file is fine).

    def mock_open(*args: Any, **kwargs: Any) -> Any:
        msg = "Simulated read error"
        raise OSError(msg)

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    with pytest.raises(ConfigurationError, match="Error reading configuration"):
        load_config(config_file)


def test_load_config_parsing_error(tmp_path: Path) -> None:
    """Test handling of YAML parsing errors."""
    config_file = tmp_path / "malformed.yaml"
    config_file.write_text("key: value: error", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        load_config(config_file)


def test_empty_config(tmp_path: Path) -> None:
    """Test loading an empty configuration file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    with pytest.raises(ConfigurationError, match="must contain a YAML dictionary"):
        load_config(config_file)


def test_extra_fields_forbidden(tmp_path: Path) -> None:
    """Test that extra fields are forbidden."""
    config_data = {
        "version": "0.1.0",
        "project": {"name": "Test", "root_dir": ".", "extra_field": "forbidden"},
        "oracle": {"dft": {"code": "vasp"}},
    }
    config_file = tmp_path / "extra.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
        load_config(config_file)
