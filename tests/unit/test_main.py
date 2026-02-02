from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.main import main


def test_main_success(temp_dir: Path) -> None:
    config_file = temp_dir / "config.yaml"
    config_file.touch()

    with (
        patch("argparse.ArgumentParser.parse_args") as mock_args,
        patch("mlip_autopipec.main.load_config") as mock_load,
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
        patch("mlip_autopipec.main.MockExplorer") as MockExp,
        patch("mlip_autopipec.main.MockOracle") as MockOra,
        patch("mlip_autopipec.main.PacemakerTrainer") as MockTra,
        patch("mlip_autopipec.main.MockValidator") as MockVal,
    ):
        mock_args.return_value.config = config_file

        # Setup config mock with minimal requirements
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        main()

        mock_load.assert_called_once_with(config_file)

        # Ensure components are instantiated
        MockExp.assert_called_once()
        MockOra.assert_called_once()
        MockTra.assert_called_once()
        MockVal.assert_called_once()

        # Ensure Orchestrator is initialized with components
        MockOrch.assert_called_once_with(
            config=mock_config,
            explorer=MockExp.return_value,
            oracle=MockOra.return_value,
            trainer=MockTra.return_value,
            validator=MockVal.return_value,
        )
        MockOrch.return_value.run.assert_called_once()


def test_main_missing_config() -> None:
    with patch("argparse.ArgumentParser.parse_args") as mock_args, patch("sys.exit") as mock_exit:
        mock_args.return_value.config = Path("ghost.yaml")
        mock_exit.side_effect = SystemExit

        with pytest.raises(SystemExit):
            main()

        mock_exit.assert_called_once_with(1)
