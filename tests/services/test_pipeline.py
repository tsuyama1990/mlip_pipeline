from pathlib import Path
from unittest.mock import MagicMock, patch

from mlip_autopipec.services.pipeline import PipelineController


def test_pipeline_execute():
    input_file = Path("input.yaml")

    with patch("mlip_autopipec.services.pipeline.ConfigFactory.from_yaml") as mock_config_factory,          patch("mlip_autopipec.services.pipeline.WorkspaceManager") as mock_workspace_cls,          patch("mlip_autopipec.services.pipeline.setup_logging") as mock_setup_logging,          patch("mlip_autopipec.services.pipeline.DatabaseManager") as mock_db_cls,          patch("mlip_autopipec.services.pipeline.WorkflowManager") as mock_workflow_manager_cls:

        mock_config = MagicMock()
        mock_config.log_path = Path("/tmp/log")
        mock_config.db_path = Path("/tmp/db")
        mock_config.working_dir = Path("/tmp/work")
        mock_config_factory.return_value = mock_config

        mock_workspace = mock_workspace_cls.return_value
        mock_db = mock_db_cls.return_value
        mock_workflow_manager = mock_workflow_manager_cls.return_value

        PipelineController.execute(input_file)

        mock_config_factory.assert_called_once_with(input_file)
        mock_workspace.setup_workspace.assert_called_once()
        mock_setup_logging.assert_called_once_with(mock_config.log_path)
        mock_db.initialize.assert_called_once()
        mock_db.set_system_config.assert_called_once_with(mock_config)

        mock_workflow_manager_cls.assert_called_once_with(mock_config, mock_config.working_dir)
        mock_workflow_manager.run.assert_called_once()
