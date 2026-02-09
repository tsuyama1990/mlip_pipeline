import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.components.trainer.activeset import ActiveSetSelector, SecurityError


@patch("subprocess.run")
def test_activeset_selector(mock_run: MagicMock, tmp_path: Path) -> None:
    # Mock subprocess success
    mock_run.return_value = MagicMock(returncode=0, stdout="Success")

    input_path = tmp_path / "input.pckl.gzip"
    input_path.touch()

    selector = ActiveSetSelector(limit=100)

    output_path = tmp_path / "filtered.pckl.gzip"

    # Simulate the tool creating the file
    def create_output(*args: Any, **kwargs: Any) -> MagicMock:
        output_path.touch()
        return MagicMock(returncode=0, stdout="Success")

    mock_run.side_effect = create_output

    with patch("shutil.which", return_value="/usr/bin/pace_activeset"):
        result_path = selector.select(input_path, output_path)

    assert result_path.exists()
    assert result_path == output_path

    # Verify command construction
    args, _ = mock_run.call_args
    command = args[0]

    assert "/usr/bin/pace_activeset" in command
    assert str(input_path.resolve()) in command
    assert "--output" in command
    assert "--max" in command  # We implemented --max


def test_activeset_selector_fail(tmp_path: Path) -> None:
    with patch("subprocess.run") as mock_run:
        # Simulate failure
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"], stderr="Error")

        selector = ActiveSetSelector(limit=100)

        # Create input file so validation passes
        input_file = tmp_path / "in.gzip"
        input_file.touch()

        # Mock shutil.which to return a trusted path
        with (
            patch("shutil.which", return_value="/usr/bin/pace_activeset"),
            pytest.raises(RuntimeError),
        ):
            selector.select(input_file, tmp_path / "out.gzip")


def test_activeset_selector_security_override(tmp_path: Path) -> None:
    selector = ActiveSetSelector(limit=100)

    # Use a path that is clearly not standard
    untrusted_bin = "/custom/bin/pace_activeset"

    # 1. Without ENV var, it should fail SecurityError
    with (
        patch("shutil.which", return_value=untrusted_bin),
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(SecurityError, match="not in a trusted directory"),
    ):
        selector._validate_executable("pace_activeset")

    # 2. With ENV var, it should pass (with warning)
    with (
        patch("shutil.which", return_value=untrusted_bin),
        patch.dict(os.environ, {"PACE_ACTIVESET_BIN": "pace_activeset"}),
    ):
        # Should not raise
        result = selector._validate_executable("pace_activeset")
        # Result is str of resolved path
        assert str(Path(untrusted_bin).resolve()) == result


def test_activeset_selector_tmp_security() -> None:
    # Test that /tmp is always rejected even with ENV var
    tmp_bin = "/tmp/pace_activeset"  # noqa: S108

    selector = ActiveSetSelector()

    with (
        patch("shutil.which", return_value=tmp_bin),
        patch.dict(os.environ, {"PACE_ACTIVESET_BIN": "pace_activeset"}),
        pytest.raises(SecurityError, match="insecure temporary directory"),
    ):
        selector._validate_executable("pace_activeset")
