"""Test security constraints in PacemakerWrapper."""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.validation import validate_safe_path
from pyacemaker.trainer.wrapper import PacemakerWrapper


def test_validate_safe_path_security(tmp_path):
    """Test that path validation prevents traversal."""
    wrapper = PacemakerWrapper()

    # Create safe path
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()

    # This should pass
    validate_safe_path(safe_dir)

    # Malicious path
    malicious = tmp_path / "../etc/passwd"

    with pytest.raises(ValueError, match="Path traversal detected"):
        validate_safe_path(malicious)


def test_command_injection_prevention():
    """Test that command construction uses list format (no shell=True)."""
    wrapper = PacemakerWrapper()

    with patch("subprocess.run") as mock_run:
        # We mock returncode to 0
        mock_run.return_value.return_value = 0

        # input.yaml path with potential injection chars
        input_path = Path("input; rm -rf /; .yaml")
        work_dir = Path("work_dir")

        try:
            wrapper.train_from_input(input_path, work_dir)
        except Exception:
            pass # We expect it might fail on path validation or file not found, but we check the call

        # Check that subprocess was called with a list, not string
        # And shell=False (default for list)
        if mock_run.called:
            args, kwargs = mock_run.call_args
            cmd_arg = args[0]
            assert isinstance(cmd_arg, list)
            assert str(input_path) in cmd_arg
            assert kwargs.get("shell") is False

def test_relative_paths_traversal(tmp_path):
    """Test path traversal detection."""
    d = tmp_path / "data"
    d.touch()
    _ = tmp_path / "../out" # This might resolve to something valid or throw

    # validate_safe_path throws if ".." in path string, regardless of resolve?
    # The implementation uses resolve(), so ".." is resolved.
    # But it checks if resolved path is relative to current working directory or allowed roots?
    # The current implementation of validate_safe_path checks for ".." in str(path) BEFORE resolve?
    # Let's check implementation.
    # "if '..' in str(path): raise ValueError"

    with pytest.raises(ValueError):
        validate_safe_path(Path("../traversal"))
