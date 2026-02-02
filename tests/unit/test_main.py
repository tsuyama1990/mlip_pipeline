from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from mlip_autopipec.main import main


def test_main_success_no_validation(temp_dir: Path) -> None:
    config_file = temp_dir / "config.yaml"
    config_file.touch()

    with (
        patch("argparse.ArgumentParser.parse_args") as mock_args,
        patch("mlip_autopipec.main.load_config") as mock_load,
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
        patch("mlip_autopipec.main.MockExplorer"),
        patch("mlip_autopipec.main.MockOracle"),
        patch("mlip_autopipec.main.MockValidator") as MockValidatorCls,
        patch("mlip_autopipec.main.PacemakerTrainer"),
    ):
        mock_args.return_value.config = config_file
        # Mock validation flag = False
        mock_load.return_value.validation.run_validation = False

        main()

        mock_load.assert_called_once_with(config_file)
        # Check that Orchestrator was called with validator=None
        MockOrch.assert_called_once_with(
            config=ANY,
            explorer=ANY,
            oracle=ANY,
            trainer=ANY,
            validator=None,
        )
        MockValidatorCls.assert_not_called()
        MockOrch.return_value.run.assert_called_once()


def test_main_success_with_validation(temp_dir: Path) -> None:
    config_file = temp_dir / "config.yaml"
    config_file.touch()

    with (
        patch("argparse.ArgumentParser.parse_args") as mock_args,
        patch("mlip_autopipec.main.load_config") as mock_load,
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
        patch("mlip_autopipec.main.MockExplorer"),
        patch("mlip_autopipec.main.MockOracle"),
        patch("mlip_autopipec.main.MockValidator") as MockValidatorCls,
        patch("mlip_autopipec.main.PacemakerTrainer"),
    ):
        mock_args.return_value.config = config_file
        # Mock validation flag = True
        mock_load.return_value.validation.run_validation = True

        main()

        mock_load.assert_called_once_with(config_file)
        # Check that Orchestrator was called with a validator
        MockOrch.assert_called_once_with(
            config=ANY,
            explorer=ANY,
            oracle=ANY,
            trainer=ANY,
            validator=MockValidatorCls.return_value,
        )
        MockValidatorCls.assert_called_once()
        MockOrch.return_value.run.assert_called_once()


def test_main_missing_config() -> None:
    with patch("argparse.ArgumentParser.parse_args") as mock_args, patch("sys.exit") as mock_exit:
        mock_args.return_value.config = Path("ghost.yaml")
        mock_exit.side_effect = SystemExit

        with pytest.raises(SystemExit):
            main()

        mock_exit.assert_called_once_with(1)
