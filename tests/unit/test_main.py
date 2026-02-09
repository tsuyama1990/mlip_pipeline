from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.constants import DEFAULT_CONFIG_PATH
from mlip_autopipec.main import main


def test_main_default_args() -> None:
    """Test main with default arguments."""
    with (
        patch("sys.argv", ["main.py"]),
        patch("mlip_autopipec.main.Orchestrator") as MockOrchestrator,
    ):
        main()

        MockOrchestrator.assert_called_once_with(DEFAULT_CONFIG_PATH)


def test_main_custom_config(tmp_path: Path) -> None:
    """Test main with custom config path."""
    config_file = tmp_path / "custom_config.yaml"

    with (
        patch("sys.argv", ["main.py", "--config", str(config_file)]),
        patch("mlip_autopipec.main.Orchestrator") as MockOrchestrator,
    ):
        main()

        MockOrchestrator.assert_called_once_with(config_file)


def test_main_exception() -> None:
    """Test that exceptions are handled and program exits with 1."""
    with (
        patch("sys.argv", ["main.py"]),
        patch("mlip_autopipec.main.Orchestrator", side_effect=ValueError("Test Error")),
        patch("sys.stderr.write") as mock_stderr,
        pytest.raises(SystemExit) as excinfo,
    ):
        main()

    assert excinfo.value.code == 1
    mock_stderr.assert_called()
    assert "Test Error" in mock_stderr.call_args[0][0]
