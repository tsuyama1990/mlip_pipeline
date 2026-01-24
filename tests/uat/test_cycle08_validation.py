from unittest.mock import patch

from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


@patch("mlip_autopipec.modules.cli_handlers.handlers.CLIHandler.validate_potential")
def test_uat_08_01_phonon_validation_cli(mock_validate):
    """
    Scenario 08-01: Phonon Stability Check
    Ensure CLI accepts --phonon flag and delegates to handler.
    """
    result = runner.invoke(app, ["validate", "config.yaml", "--phonon"])

    assert result.exit_code == 0
    mock_validate.assert_called_once()
    args, kwargs = mock_validate.call_args
    assert str(args[0]) == "config.yaml"
    assert args[1]["phonon"] is True


def test_uat_08_03_cli_usability():
    """
    Scenario 08-03: CLI Usability
    Ensure help message is clear and correct.
    """
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "Validate the potential" in result.stdout
    assert "--phonon" in result.stdout
    assert "--elastic" in result.stdout
    assert "--eos" in result.stdout


def test_uat_08_02_e2e_command_check():
    """
    Scenario 08-02: End-to-End Run Command
    Verify run loop command exists.
    """
    result = runner.invoke(app, ["run", "loop", "--help"])
    assert result.exit_code == 0
    assert "Run the full autonomous loop" in result.stdout
