import pytest
import sys
from unittest.mock import patch
from mlip_autopipec.main import main


def test_main_no_args(capsys):
    with patch.object(sys, "argv", ["mlip-run"]):
        with pytest.raises(SystemExit) as e:
            main()
    assert e.type is SystemExit
    assert e.value.code == 2


def test_main_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.touch()

    with patch.object(sys, "argv", ["mlip-run", str(config_file)]), \
         patch("mlip_autopipec.main.Orchestrator") as mock_orch:

        main()

        mock_orch.assert_called_once()
        mock_orch.return_value.run_cycle.assert_called_once()


def test_main_missing_config(tmp_path, capsys):
    config_file = tmp_path / "missing.yaml"

    with patch.object(sys, "argv", ["mlip-run", str(config_file)]), \
         pytest.raises(SystemExit) as e:
        main()

    assert e.value.code == 1
    out, err = capsys.readouterr()
    assert "Config file not found" in err


def test_main_orchestrator_error(tmp_path, capsys):
    config_file = tmp_path / "config.yaml"
    config_file.touch()

    with patch.object(sys, "argv", ["mlip-run", str(config_file)]), \
         patch("mlip_autopipec.main.Orchestrator") as mock_orch, \
         pytest.raises(SystemExit) as e:

        mock_orch.side_effect = Exception("Setup failed")
        main()

    assert e.value.code == 1
    out, err = capsys.readouterr()
    assert "Error running pipeline: Setup failed" in err
