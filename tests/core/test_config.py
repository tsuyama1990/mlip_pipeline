"""Tests for configuration management."""

import os
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
        DFTConfig(
            code="vasp",
            parameters={1: "invalid"},  # type: ignore[dict-item]
            pseudopotentials={"Fe": "Fe.pbe.UPF"},
        )
    # Pydantic's default validation message for incorrect dict key type
    assert "Input should be a valid string" in str(excinfo.value)


def test_dft_config_pseudopotentials_missing(tmp_path: Path) -> None:
    """Test DFTConfig pseudopotentials validation."""
    # Temporarily disable skip_file_checks if it was enabled globally
    original_skip = CONSTANTS.skip_file_checks
    CONSTANTS.skip_file_checks = False
    try:
        with pytest.raises(ValidationError, match="Missing pseudopotential"):
            DFTConfig(
                code="quantum_espresso",
                pseudopotentials={"Fe": "non_existent.UPF"},
            )
    finally:
        CONSTANTS.skip_file_checks = original_skip


def test_dft_config_pseudopotentials_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DFTConfig pseudopotentials validation with valid files."""
    # Ensure checks are enabled but run in correct dir
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", False)
    monkeypatch.chdir(tmp_path)

    pp_file = tmp_path / "Fe.pbe.UPF"
    pp_file.touch()

    # Use relative path since we changed CWD
    rel_pp_file = "Fe.pbe.UPF"

    config = DFTConfig(
        code="quantum_espresso",
        pseudopotentials={"Fe": rel_pp_file},
    )
    assert config.pseudopotentials["Fe"] == rel_pp_file


def test_version_validation() -> None:
    """Test semantic version validation."""
    original_skip = CONSTANTS.skip_file_checks
    CONSTANTS.skip_file_checks = True
    try:
        data = {
            "version": "invalid",
            "project": {"name": "Test", "root_dir": "."},
            "oracle": {"dft": {"code": "vasp", "pseudopotentials": {"Fe": "Fe.pbe.UPF"}}},
        }
        with pytest.raises(ValidationError) as excinfo:
            PYACEMAKERConfig(**data)  # type: ignore[arg-type]
        # Pydantic's regex mismatch message
        assert "String should match pattern" in str(excinfo.value)
    finally:
        CONSTANTS.skip_file_checks = original_skip


def test_load_config_file_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a file that exceeds the size limit."""
    # Enable file checks explicitly to test size check logic (though size check is independent of security check)
    # Wait, size check is in _read_file_content which is called by load_config.
    # _validate_file_security is called BEFORE _read_file_content.
    # So we MUST skip security check to reach size check (since path is in tmp), OR chdir.
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)

    config_file = tmp_path / "large.yaml"
    config_file.touch()

    # Mock stat to return a large size

    class MockStat:
        st_size = CONSTANTS.max_config_size + 1
        st_mode = 33188  # Regular file mode
        st_uid = os.getuid()

    # Fix: Accept **kwargs to handle follow_symlinks argument passed by pathlib/pytest
    monkeypatch.setattr("pathlib.Path.stat", lambda self, **kwargs: MockStat())

    with pytest.raises(ConfigurationError, match="Configuration file too large"):
        load_config(config_file)


def test_load_config_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of OSError during file read."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)

    config_file = tmp_path / "valid.yaml"
    config_file.touch()

    # We mock open to raise OSError.
    # We allow stat to pass (default behavior on touched file is fine).

    def mock_open(*args: Any, **kwargs: Any) -> Any:
        msg = "Simulated read error"
        raise OSError(msg)

    # Mock os.access
    monkeypatch.setattr(os, "access", lambda path, mode: True)
    monkeypatch.setattr("pathlib.Path.open", mock_open)

    with pytest.raises(ConfigurationError, match="Unexpected error reading configuration"):
        load_config(config_file)


def test_load_config_parsing_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of YAML parsing errors."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    config_file = tmp_path / "malformed.yaml"
    config_file.write_text("key: value: error", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        load_config(config_file)


def test_empty_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading an empty configuration file."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    with pytest.raises(ConfigurationError, match="must contain a YAML dictionary"):
        load_config(config_file)


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
