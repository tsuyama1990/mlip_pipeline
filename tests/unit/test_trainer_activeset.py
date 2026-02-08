import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.components.trainer.activeset import ActiveSetSelector


@patch("subprocess.run")
def test_activeset_selector(mock_run, tmp_path: Path):
    # Mock subprocess success
    mock_run.return_value = MagicMock(returncode=0, stdout="Success")

    input_path = tmp_path / "input.pckl.gzip"
    input_path.touch()

    selector = ActiveSetSelector(limit=100)

    output_path = tmp_path / "filtered.pckl.gzip"

    # Simulate the tool creating the file
    def create_output(*args, **kwargs):
        output_path.touch()
        return MagicMock(returncode=0, stdout="Success")

    mock_run.side_effect = create_output

    result_path = selector.select(input_path, output_path)

    assert result_path.exists()
    assert result_path == output_path

    # Verify command construction
    args, _ = mock_run.call_args
    command = args[0]

    assert "pace_activeset" in command
    assert str(input_path) in command
    assert "--output" in command
    assert "--max" in command  # We implemented --max

def test_activeset_selector_fail(tmp_path):
    with patch("subprocess.run") as mock_run:
        # Simulate failure
        mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"], stderr="Error")

        selector = ActiveSetSelector(limit=100)

        with pytest.raises(RuntimeError):
            selector.select(tmp_path / "in.gzip", tmp_path / "out.gzip")
