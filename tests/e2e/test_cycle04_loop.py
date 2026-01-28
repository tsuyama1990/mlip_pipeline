from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config.models import UserInputConfig, WorkflowConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.domain_models.state import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def mock_config(tmp_path):
    # Create fake UPF file
    (tmp_path / "Al.UPF").touch()

    return UserInputConfig(
        target_system=TargetSystem(elements=["Al"], crystal_structure="fcc", composition={"Al": 1}),
        dft=DFTConfig(
            qs_executable="pw.x",
            pseudopotential_dir=tmp_path,
            pseudopotentials={"Al": "Al.UPF"},
            k_points=[1,1,1]
        ),
        workflow_config=WorkflowConfig(max_generations=3)
    )


def test_workflow_resume(mock_config, tmp_path):
    # Setup state file: Cycle 1, Phase Calculation (DFT)
    state = WorkflowState(
        cycle_index=1,
        current_phase=WorkflowPhase.CALCULATION,
        latest_potential_path=Path("pot.yace")
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(state.model_dump_json())

    manager = WorkflowManager(mock_config, tmp_path)

    # Mock Phases
    with patch("mlip_autopipec.orchestration.workflow.ExplorationPhase") as MockExploration, \
         patch("mlip_autopipec.orchestration.workflow.SelectionPhase"), \
         patch("mlip_autopipec.orchestration.workflow.DFTPhase") as MockDFT, \
         patch("mlip_autopipec.orchestration.workflow.TrainingPhase") as MockTraining:

        manager.run()

        # Cycle 1:
        # Should SKIP Exploration and Selection
        # Should RUN DFT and Training

        # Cycle 2:
        # Should RUN ALL

        # Verifications
        # Exploration: Called once (for Cycle 2), NOT for Cycle 1
        # The phase class is instantiated and then execute() is called.
        # So we check instantiation count.

        # Wait, if we resume from CALCULATION phase of Cycle 1.
        # Cycle 1: Skip Exp, Skip Sel, Run Calc, Run Train
        # Cycle 2: Run Exp, Run Sel, Run Calc, Run Train

        # Exploration: Called once (Cycle 2)
        assert MockExploration.call_count == 1

        # DFT: Called twice (Cycle 1, Cycle 2)
        assert MockDFT.call_count == 2

        # Training: Called twice
        assert MockTraining.call_count == 2
