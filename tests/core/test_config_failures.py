"""Tests for configuration failure scenarios."""

import io
import stat
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from pyacemaker.core.config import CONSTANTS, load_config
from pyacemaker.core.exceptions import ConfigurationError


def test_load_config_not_found() -> None:
    """Test loading a non-existent file."""
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(Path("non_existent.yaml"))


def test_load_config_permission_error(tmp_path: Path) -> None:
    """Test loading a file with permission error."""
    # Use a real path but mock os.access
    config_file = tmp_path / "config.yaml"
    config_file.touch()

    # Mock os.access to return False
    with (
        patch("os.access", return_value=False),
        pytest.raises(ConfigurationError, match="Permission denied")
    ):
        load_config(config_file)


def test_load_config_too_large_stat(tmp_path: Path) -> None:
    """Test loading a file that stat reports as too large."""
    config_file = tmp_path / "large.yaml"
    config_file.touch()

    # Mock os.access to return True
    # Mock Path.stat to return large size
    # We must ensure st_mode indicates a regular file for is_file() to work
    with (
        patch("os.access", return_value=True),
        patch.object(Path, "stat") as mock_stat
    ):
        mock_stat.return_value.st_size = CONSTANTS.max_config_size + 1
        mock_stat.return_value.st_mode = stat.S_IFREG

        with pytest.raises(ConfigurationError, match="Configuration file too large"):
            load_config(config_file)


def test_load_config_too_large_stream(tmp_path: Path) -> None:
    """Test loading a file that grows beyond limit during reading."""
    config_file = tmp_path / "growing.yaml"
    config_file.touch()

    # Create large content
    content = "key: value\n" * (CONSTANTS.max_config_size // 10 + 100)

    # Mock os.access to return True
    # Mock Path.stat to return small size
    # Mock Path.open to return large content
    with (
        patch("os.access", return_value=True),
        patch.object(Path, "stat") as mock_stat,
        patch.object(Path, "open") as mock_open
    ):
        mock_stat.return_value.st_size = 100
        mock_stat.return_value.st_mode = stat.S_IFREG

        mock_file = Mock()
        # Simulate file reading
        mock_file.__enter__ = Mock(return_value=io.StringIO(content))
        mock_file.__exit__ = Mock(return_value=None)
        mock_open.return_value = mock_file

        with pytest.raises(ConfigurationError, match="exceeds limit"):
            load_config(config_file)


def test_load_config_yaml_error(tmp_path: Path) -> None:
    """Test loading invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("key: value: error")

    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        load_config(config_file)


def test_load_config_validation_error(tmp_path: Path) -> None:
    """Test loading valid YAML but invalid schema."""
    config_file = tmp_path / "invalid_schema.yaml"
    data = {"version": "invalid_ver"}
    config_file.write_text(yaml.dump(data))

    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(config_file)
