"""Tests for Pacemaker Wrapper Security."""

from pathlib import Path

import pytest

from pyacemaker.trainer.wrapper import PacemakerWrapper


def test_sanitize_arg_valid() -> None:
    wrapper = PacemakerWrapper()
    assert wrapper._sanitize_arg("cutoff", 5.0) == ["--cutoff", "5.0"]
    assert wrapper._sanitize_arg("some_flag", True) == ["--some-flag"]
    assert wrapper._sanitize_arg("list", [1, 2]) == ["--list", "1", "2"]


def test_sanitize_arg_invalid_key() -> None:
    wrapper = PacemakerWrapper()
    with pytest.raises(ValueError, match="Invalid parameter key"):
        wrapper._sanitize_arg("invalid key", 1)
    with pytest.raises(ValueError, match="Invalid parameter key"):
        wrapper._sanitize_arg("key;rm", 1)


def test_sanitize_arg_invalid_value() -> None:
    wrapper = PacemakerWrapper()
    # Control chars
    with pytest.raises(ValueError, match="Invalid control characters"):
        wrapper._sanitize_arg("key", "val\nue")

    # Check if we allowed other chars like ;
    # Regex in wrapper.py: if re.search(r"[\x00-\x1f]", val_str):
    # It allows ; because subprocess.run(shell=False) is used.
    # But let's check what I implemented in the file write earlier.
    # I verified against control chars.

    # Semicolon should be passed as literal argument to the program.
    # So ["--key", "val;ue"] is safe with shell=False.
    assert wrapper._sanitize_arg("key", "val;ue") == ["--key", "val;ue"]


def test_validate_paths(tmp_path: Path) -> None:
    wrapper = PacemakerWrapper()
    d = tmp_path / "data"
    d.touch()
    o = tmp_path / "out"

    wrapper._validate_paths(d, o)
    assert o.exists()

def test_validate_paths_traversal(tmp_path: Path) -> None:
    wrapper = PacemakerWrapper()
    d = tmp_path / "data"
    d.touch()
    o = tmp_path / "../out" # This might resolve to something valid or throw

    # validate_safe_path throws if ".." in path string, regardless of resolve?
    # Let's verify behavior.
    with pytest.raises(ValueError, match="Path traversal"):
        wrapper._validate_paths(d, Path("../out"))
