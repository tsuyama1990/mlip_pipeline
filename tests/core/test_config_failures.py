"""Tests for configuration failure modes."""

import stat
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from pyacemaker.core.config import CONSTANTS, load_config
from pyacemaker.core.exceptions import ConfigurationError


def test_load_config_not_found(tmp_path: Path) -> None:
    """Test loading a non-existent configuration file."""
    config_file = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(config_file)


def test_load_config_permission_error() -> None:
    """Test loading a configuration file without read permissions."""
    # Use a Mock object for path to avoid creating actual files/permissions
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.name = "protected.yaml"

    # Patch os.access to simulate permission denied
    with (
        patch("os.access", return_value=False),
        pytest.raises(ConfigurationError, match="Permission denied"),
    ):
        load_config(mock_path)


def test_load_config_path_traversal() -> None:
    """Test path traversal detection in ProjectConfig."""
    from pyacemaker.core.config import ProjectConfig

    # Direct traversal attempt
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        ProjectConfig(name="test", root_dir=Path("../../../etc/passwd"))

    # Traversal in parts
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        ProjectConfig(name="test", root_dir=Path("safe/../../unsafe"))


def test_load_config_too_large_stat() -> None:
    """Test loading a configuration file that reports too large size in stat."""
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.name = "large.yaml"

    # Configure stat size
    mock_stat = MagicMock()
    mock_stat.st_size = CONSTANTS.max_config_size + 1
    # Mock file ownership check
    import os

    mock_stat.st_uid = os.getuid()
    # Ensure st_mode is an integer compatible with bitwise operations
    # Ensure st_mode is an integer compatible with bitwise operations
    mock_stat.st_mode = 33188

    # Mock return value of resolve()
    mock_path.resolve.return_value = mock_path
    mock_path.stat.return_value = mock_stat

    # Patch os.access to allow read
    with (
        patch("os.access", return_value=True),
        pytest.raises(ConfigurationError, match="Configuration file too large"),
    ):
        load_config(mock_path)


def test_load_config_too_large_stream(tmp_path: Path) -> None:
    """Test loading a configuration file that exceeds size limit during read."""
    # This tests LimitedStream behavior via load_config
    config_file = tmp_path / "stream_large.yaml"

    # Create a file slightly larger than limit
    # We create a valid YAML structure but repeat it to exceed size
    content = "key: value\n" * (CONSTANTS.max_config_size // 10 + 100)
    config_file.write_text(content)

    # Patch stat size to be small so it passes the initial check
    # But reading it will fail
    import os

    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 100
        # Must also mock st_mode for is_file check if it calls stat
        mock_stat.return_value.st_mode = stat.S_IFREG
        mock_stat.return_value.st_uid = os.getuid()

        # We need to ensure is_file returns True, which might depend on stat
        # Or we can patch is_file on the specific path object? No, Path methods are on class.
        # But we are using a real path object here.
        # Let's try to patch os.access as well just in case
        with (
            patch("os.access", return_value=True),
            # Check for both possible error messages depending on where it's caught
            # But implementation raises "Configuration file exceeds limit" from ValueError
            pytest.raises(ConfigurationError, match="exceeds limit"),
        ):
            load_config(config_file)


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Test loading an invalid YAML file."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("key: value: invalid\n  indentation_error")

    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        load_config(config_file)


def test_load_config_corrupted_stream(tmp_path: Path) -> None:
    """Test loading a file that causes read errors (e.g. encoding)."""
    config_file = tmp_path / "corrupt.yaml"
    # Write invalid utf-8 byte sequence
    with config_file.open("wb") as f:
        f.write(b"\x80\x81\xff")

    # This should raise UnicodeDecodeError during read, caught as Unexpected error
    with pytest.raises(ConfigurationError, match="Unexpected error"):
        load_config(config_file)


def test_load_config_not_dict(tmp_path: Path) -> None:
    """Test loading a valid YAML file that is not a dictionary."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2")

    with pytest.raises(ConfigurationError, match="must contain a YAML dictionary"):
        load_config(config_file)


def test_load_config_validation_error(tmp_path: Path) -> None:
    """Test loading a YAML file that fails Pydantic validation."""
    config_file = tmp_path / "invalid_schema.yaml"
    # Missing required 'project' and 'oracle' fields
    config_file.write_text("version: 0.1.0\nlogging:\n  level: INFO")

    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(config_file)
