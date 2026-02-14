"""Tests for configuration security validation."""

import pytest

from pyacemaker.core.config import _recursive_validate_parameters


def test_validate_parameters_whitelist_keys() -> None:
    """Test that keys are validated against whitelist."""
    # Valid keys
    valid_data = {"key_1": "value", "sub-key": "value", "key.name": "value"}
    _recursive_validate_parameters(valid_data)

    # Invalid keys
    with pytest.raises(ValueError, match="Invalid characters in key"):
        _recursive_validate_parameters({"key with space": "value"})

    with pytest.raises(ValueError, match="Invalid characters in key"):
        _recursive_validate_parameters({"key$": "value"})


def test_validate_parameters_whitelist_values() -> None:
    """Test that values are validated against whitelist."""
    # Valid values
    valid_data = {"key": "value", "path": "/path/to/file.txt", "list": [1, 2]}
    _recursive_validate_parameters(valid_data)

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
            _recursive_validate_parameters({"key": val})


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
        _recursive_validate_parameters(deep_data)
