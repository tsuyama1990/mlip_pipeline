from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.main import main


def test_main_success(temp_dir: Path) -> None:
    config_file = temp_dir / "config.yaml"
    config_file.touch()

    with (
        patch("argparse.ArgumentParser.parse_args") as mock_args,
        patch("mlip_autopipec.main.load_config") as mock_load,
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
    ):
        mock_args.return_value.config = config_file

        main()

        mock_load.assert_called_once_with(config_file)
        MockOrch.assert_called_once()
        MockOrch.return_value.run.assert_called_once()


def test_main_missing_config() -> None:
    with patch("argparse.ArgumentParser.parse_args") as mock_args, patch("sys.exit") as mock_exit:
        mock_args.return_value.config = Path("ghost.yaml")
        mock_exit.side_effect = SystemExit

        with pytest.raises(SystemExit):
            main()

        mock_exit.assert_called_once_with(1)
