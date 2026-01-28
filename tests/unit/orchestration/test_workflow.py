from unittest.mock import patch

import pytest

from mlip_autopipec.config.models import (
    MLIPConfig,
    TargetSystem,
    WorkflowConfig,
)
from mlip_autopipec.config.schemas.core import RuntimeConfig
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.domain_models.state import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config(tmp_path):
    # Create fake UPF file
    (tmp_path / "Fe.UPF").touch()

    return MLIPConfig(
        target_system=TargetSystem(name="Test", elements=["Fe"], composition={"Fe": 1.0}),
        dft=DFTConfig(
            pseudopotential_dir=tmp_path, ecutwfc=30, kspacing=0.05, command="pw.x"
        ),
        workflow=WorkflowConfig(max_generations=2),
        runtime=RuntimeConfig(work_dir=tmp_path / "_work", database_path="mlip.db"),
    )


@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_workflow_manager_initialization(mock_db, mock_config):
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)
    assert isinstance(manager.state, WorkflowState)
    assert manager.state.cycle_index == 0
    # Current phase default is EXPLORATION?
    # WorkflowState defaults?
    # Check WorkflowState definition if needed. Assuming defaults.


@patch("mlip_autopipec.orchestration.workflow.DatabaseManager")
def test_workflow_run_loop_cycle_02(mock_db, mock_config):
    # Test run_cycle_02 logic roughly
    manager = WorkflowManager(mock_config, work_dir=mock_config.runtime.work_dir)

    # We mock StructureBuilder and QERunner inside run_cycle_02 usually, or use mock_dft=True
    # To test run_cycle_02 execution path:

    with patch("mlip_autopipec.orchestration.workflow.StructureBuilder") as mock_builder:
        mock_builder.return_value.build.return_value = []

        manager.run_cycle_02(mock_dft=True, dry_run=True)

        # Assertions
        mock_builder.assert_called()
