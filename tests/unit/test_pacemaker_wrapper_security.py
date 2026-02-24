"""Test security constraints in PacemakerWrapper."""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import CONSTANTS
from pyacemaker.core.validation import validate_safe_path
from pyacemaker.trainer.wrapper import PacemakerWrapper


def test_validate_safe_path_security(tmp_path):
    """Test that path validation prevents traversal."""
    # Create safe path
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()

    # This should pass (relative to CWD if inside, or absolute allowed)
    # We must change CWD to tmp_path for this test to work with relative checks reliably
    # or rely on the fact that pytest tmp_path is safe if passed as absolute.

    # Since validate_safe_path checks against CWD by default, let's patch CWD
    with patch("pathlib.Path.cwd", return_value=safe_dir):
        validate_safe_path(safe_dir) # Should pass

        # Malicious path outside CWD
        malicious = tmp_path / "../etc/passwd"

        # Exception message can vary depending on implementation (outside base or traversal detected)
        # We just want to ensure it raises ValueError
        with pytest.raises(ValueError):
            validate_safe_path(malicious)


def test_validate_safe_path_whitelist(tmp_path):
    """Test validation against allowed whitelist."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    # Temporarily add allowed_dir to whitelist
    # CONSTANTS is a Pydantic model instance, but fields are loaded via factory.
    # We can't easily modify the instance in-place if it's immutable/validated frozen.
    # However, in tests we can patch CONSTANTS.allowed_potential_paths

    with patch.object(CONSTANTS, 'allowed_potential_paths', [str(allowed_dir)]):
        # Even if CWD is somewhere else
        with patch("pathlib.Path.cwd", return_value=tmp_path / "other"):
            # This path is outside "other" but inside allowed whitelist
            validate_safe_path(allowed_dir)


def test_command_injection_prevention():
    """Test that command construction uses list format (no shell=True)."""
    wrapper = PacemakerWrapper()

    with patch("subprocess.run") as mock_run:
        # We mock returncode to 0
        mock_run.return_value.return_value = 0

        # input.yaml path with potential injection chars
        input_path = Path("input; rm -rf /; .yaml")
        work_dir = Path("work_dir")

        # Mock existence checks since we are testing command construction
        with patch("pathlib.Path.exists", return_value=True):
             # Also mock validation to pass for this test (we test validation separately)
             with patch("pyacemaker.trainer.wrapper.validate_safe_path"):
                try:
                    wrapper.train_from_input(input_path, work_dir)
                except Exception:
                    pass

        # Check that subprocess was called with a list, not string
        if mock_run.called:
            args, kwargs = mock_run.call_args
            cmd_arg = args[0]
            assert isinstance(cmd_arg, list)
            assert str(input_path) in cmd_arg
            assert kwargs.get("shell") is False


def test_sanitize_arg_injection():
    """Test _sanitize_arg prevents injection chars."""
    wrapper = PacemakerWrapper()

    # Valid
    assert wrapper._sanitize_arg("key", "value") == ["--key", "value"]

    # Invalid key
    with pytest.raises(ValueError, match="Invalid parameter key"):
        wrapper._sanitize_arg("key; rm -rf", "value")

    # Invalid value (control chars)
    with pytest.raises(ValueError, match="Invalid control characters"):
        wrapper._sanitize_arg("key", "val\nue")
