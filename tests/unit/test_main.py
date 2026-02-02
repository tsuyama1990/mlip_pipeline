import pytest
from unittest.mock import patch
from mlip_autopipec.main import main
import sys

def test_main_success(valid_config_yaml):
    with patch("sys.argv", ["main", str(valid_config_yaml)]), \
         patch("mlip_autopipec.main.Orchestrator") as MockOrch:

        main()
        MockOrch.assert_called_once()
        MockOrch.return_value.run.assert_called_once()

def test_main_no_config():
    with patch("sys.argv", ["main", "ghost.yaml"]), \
         patch("sys.stderr") as mock_stderr, \
         pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1

def test_main_exception(valid_config_yaml):
    with patch("sys.argv", ["main", str(valid_config_yaml)]), \
         patch("mlip_autopipec.main.Orchestrator") as MockOrch, \
         patch("sys.stderr") as mock_stderr, \
         pytest.raises(SystemExit) as exc:

        MockOrch.side_effect = Exception("Boom")
        main()

    assert exc.value.code == 1
    mock_stderr.write.assert_called()
