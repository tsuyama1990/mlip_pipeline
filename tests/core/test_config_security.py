"""Tests for configuration security validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyacemaker.core.config import CONSTANTS, DFTConfig, _validate_structure


def test_validate_parameters_whitelist_keys() -> None:
    """Test that keys are validated against whitelist."""
    # Valid keys
    valid_data = {"key_1": "value", "sub-key": "value", "key.name": "value"}
    _validate_structure(valid_data)

    # Invalid keys
    with pytest.raises(ValueError, match="Invalid characters in key"):
        _validate_structure({"key with space": "value"})

    with pytest.raises(ValueError, match="Invalid characters in key"):
        _validate_structure({"key$": "value"})


def test_validate_parameters_whitelist_values() -> None:
    """Test that values are validated against whitelist."""
    # Valid values
    valid_data = {"key": "value", "path": "/path/to/file.txt", "list": [1, 2]}
    _validate_structure(valid_data)

    # Invalid values (Shell injection attempts)
    invalid_values = [
        "value; rm -rf /",
        "$(whoami)",
        "`ls`",
        "value | cat",
        "value && ls",
        "value || echo",
        "value > file",
        "value < file",
    ]
    for val in invalid_values:
        with pytest.raises(ValueError, match="Invalid characters in value"):
            _validate_structure({"key": val})


def test_validate_parameters_depth_limit() -> None:
    """Test recursion depth limit."""
    # Construct a deep dictionary programmatically to be sure
    # Use Any to bypass mypy strict checking for this dynamic construction test
    from typing import Any

    deep_data: dict[str, Any] = {"level": 0}
    current = deep_data
    for i in range(15):
        current["next"] = {"level": i + 1}
        current = current["next"]

    with pytest.raises(ValueError, match="Configuration nesting too deep"):
        _validate_structure(deep_data)

def test_dft_pseudopotentials_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test path traversal check in DFT pseudopotentials."""
    # Re-enable security checks explicitly for this test
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", False)

    # We need a file that exists outside CWD to test the security check.
    # But for test isolation, we should use tmp_path.
    # We can mock Path.cwd to be inside tmp_path/safe, and try to access tmp_path/unsafe.

    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()

    pp_file = unsafe_dir / "evil.upf"
    pp_file.touch()

    monkeypatch.chdir(safe_dir)

    # Path traversal to unsafe dir
    rel_path = "../unsafe/evil.upf"

    with pytest.raises(ValidationError, match="outside allowed base directory"):
        DFTConfig(
            code="qe",
            pseudopotentials={"Fe": rel_path}
        )
