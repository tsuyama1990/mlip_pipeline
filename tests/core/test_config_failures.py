"""Tests for configuration loading failures."""

import io
from pathlib import Path
from unittest.mock import Mock

import pytest

from pyacemaker.core.config import CONSTANTS, load_config
from pyacemaker.core.exceptions import ConfigurationError


def test_load_config_file_not_found(tmp_path: Path) -> None:
    """Test loading a non-existent configuration file."""
    config_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(config_file)


def test_load_config_not_a_file(tmp_path: Path) -> None:
    """Test loading a directory as configuration file."""
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()

    # Use a Mock that behaves like a directory Path but is accepted by load_config checks
    with pytest.raises(ConfigurationError, match="Configuration file not found or invalid"):
        load_config(config_dir)


def test_load_config_permission_error(tmp_path: Path) -> None:
    """Test loading a file with permission error."""
    # Use a Mock path to simulate PermissionError on open
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.stat.return_value.st_size = 100
    mock_path.name = "config.yaml"
    mock_path.open.side_effect = PermissionError("Permission denied")

    with pytest.raises(ConfigurationError, match="Error reading configuration file"):
        load_config(mock_path)


def test_load_config_too_large_stat() -> None:
    """Test loading a file that stat reports as too large."""
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.stat.return_value.st_size = CONSTANTS.max_config_size + 1
    mock_path.name = "large.yaml"

    with pytest.raises(ConfigurationError, match="Configuration file too large"):
        load_config(mock_path)


def test_load_config_too_large_stream() -> None:
    """Test loading a file that grows beyond limit during reading."""
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.stat.return_value.st_size = 100  # Report small size
    mock_path.name = "growing.yaml"

    # Create large content
    content = "key: value\n" * (CONSTANTS.max_config_size // 10 + 100)

    # Configure mock open context manager
    mock_file = Mock()
    mock_file.__enter__ = Mock(return_value=io.StringIO(content))
    mock_file.__exit__ = Mock(return_value=None)
    mock_path.open.return_value = mock_file

    with pytest.raises(ConfigurationError, match="Configuration file exceeds limit"):
        load_config(mock_path)


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Test loading invalid YAML content."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("key: value: invalid")  # Invalid syntax

    with pytest.raises(ConfigurationError, match="Error parsing YAML configuration"):
        load_config(config_file)


def test_load_config_not_dict(tmp_path: Path) -> None:
    """Test loading valid YAML that is not a dictionary."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2")

    with pytest.raises(
        ConfigurationError, match="Configuration file must contain a YAML dictionary"
    ):
        load_config(config_file)


def test_validate_root_dir_traversal() -> None:
    """Test root directory validation for path traversal."""
    from pyacemaker.core.config import ProjectConfig

    # If we use a real path that doesn't exist:
    bad_path = Path("/non_existent/../path")

    with pytest.raises(ValueError, match="Path traversal not allowed"):
        ProjectConfig.validate_root_dir(bad_path)

    # Test valid absolute path (even if not exists, but no "..")
    good_path = Path("/non_existent/path")
    # This should pass validation (returns absolute)
    assert ProjectConfig.validate_root_dir(good_path) == good_path.absolute()
