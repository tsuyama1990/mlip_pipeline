from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()

@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
@patch("mlip_autopipec.orchestration.workflow.WorkflowManager.run")
@patch("mlip_autopipec.modules.cli_handlers.handlers.load_config")
def test_cli_run_cycle_02(mock_load, mock_run, mock_db, mock_tq, tmp_path):
    # Setup
    config_file = tmp_path / "input.yaml"
    config_file.touch()

    # Mock config object
    mock_conf_obj = MagicMock()
    mock_conf_obj.runtime.work_dir = tmp_path / "_work"
    # Ensure workflow_config can be assigned
    mock_conf_obj.workflow_config = None
    mock_load.return_value = mock_conf_obj

    result = runner.invoke(app, ["run-cycle-02", "--config", str(config_file)])

    assert result.exit_code == 0
    assert "Cycle 02 Pipeline Completed" in result.stdout

    # Check max_generations set to 1
    # Since we assigned a real WorkflowConfig object to the mock property
    assert mock_conf_obj.workflow_config.max_generations == 1

    # Check run called
    mock_run.assert_called_once()


@patch("mlip_autopipec.orchestration.phases.exploration.ExplorationPhase.execute")
@patch("mlip_autopipec.modules.cli_handlers.handlers.load_config")
@patch("mlip_autopipec.orchestration.workflow.TaskQueue")
@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_cli_run_cycle_02_dry_run(mock_db, mock_tq, mock_load, mock_explore, tmp_path):
    # Setup
    config_file = tmp_path / "input.yaml"
    config_file.touch()

    mock_conf_obj = MagicMock()
    mock_conf_obj.runtime.work_dir = tmp_path / "_work"
    mock_conf_obj.workflow_config = None
    mock_load.return_value = mock_conf_obj

    result = runner.invoke(app, ["run-cycle-02", "--config", str(config_file), "--dry-run"])

    assert result.exit_code == 0
    assert "Dry Run" in result.stdout
    mock_explore.assert_called_once()
