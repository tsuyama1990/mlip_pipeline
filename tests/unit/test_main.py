from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.main import main


def test_main_success(valid_config_yaml: Path) -> None:
    with patch("sys.argv", ["main", str(valid_config_yaml)]), \
         patch("mlip_autopipec.main.Orchestrator") as MockOrch:

        main()
        MockOrch.assert_called_once()
        MockOrch.return_value.run.assert_called_once()


def test_main_no_config() -> None:
    with patch("sys.argv", ["main", "ghost.yaml"]), \
         patch("sys.stderr"), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_exception(valid_config_yaml: Path) -> None:
    with patch("sys.argv", ["main", str(valid_config_yaml)]), \
         patch("mlip_autopipec.main.Orchestrator") as MockOrch, \
         patch("sys.stderr") as mock_stderr:

        MockOrch.side_effect = Exception("Boom")
        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
        mock_stderr.write.assert_called()
