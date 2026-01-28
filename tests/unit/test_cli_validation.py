from unittest.mock import patch

from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


@patch("mlip_autopipec.modules.cli_handlers.handlers.CLIHandler.validate_config")
def test_validate_config_mode(mock_validate_config):
    # Default behavior: config validation
    # Must use --config option
    result = runner.invoke(app, ["validate", "--config", "input.yaml"])
    assert result.exit_code == 0
    mock_validate_config.assert_called_once()


@patch("mlip_autopipec.modules.cli_handlers.handlers.CLIHandler.run_physics_validation")
def test_validate_physics_mode_phonon(mock_physics_validation):
    # With flags: physics validation
    result = runner.invoke(app, ["validate", "--config", "input.yaml", "--phonon"])
    assert result.exit_code == 0
    mock_physics_validation.assert_called_once()
    # Check arguments
    args, kwargs = mock_physics_validation.call_args
    # The first arg should be the config file path (Path object)
    assert str(args[0]) == "input.yaml"
    assert kwargs["phonon"] is True
    assert kwargs["elastic"] is False
    assert kwargs["eos"] is False


@patch("mlip_autopipec.modules.cli_handlers.handlers.CLIHandler.run_physics_validation")
def test_validate_physics_mode_all(mock_physics_validation):
    result = runner.invoke(app, ["validate", "--config", "input.yaml", "--phonon", "--elastic", "--eos"])
    assert result.exit_code == 0
    args, kwargs = mock_physics_validation.call_args
    assert kwargs["phonon"] is True
    assert kwargs["elastic"] is True
    assert kwargs["eos"] is True
