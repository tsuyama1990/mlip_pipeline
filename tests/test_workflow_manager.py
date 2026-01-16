"""Unit tests for the WorkflowManager module."""

from unittest.mock import patch
from mlip_autopipec.workflow_manager import WorkflowManager
from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import UserInputConfig


def test_workflow_manager_run():
    """Test the main run method of the WorkflowManager."""
    user_config_dict = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Ni"],
            "composition": {"Ni": 1.0},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
    }
    user_config = UserInputConfig.model_validate(user_config_dict)
    system_config = ConfigFactory.from_user_input(user_config)

    manager = WorkflowManager(system_config)

    # We don't have the other modules implemented yet, so we just check
    # that the run method executes without error.
    # In the future, we would mock the modules and assert they are called
    # in the correct order.
    with patch("builtins.print") as mock_print:
        manager.run()
        assert mock_print.call_count > 0
