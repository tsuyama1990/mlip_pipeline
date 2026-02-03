from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config import (
    Config,
    DFTConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    StructureGenConfig,
    TrainingConfig,
    ValidationConfig,
)
from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.orchestrator import Orchestrator


@pytest.fixture
def mock_config(temp_dir: Path) -> Config:
    (temp_dir / "data.pckl").touch()
    return Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=temp_dir / "data.pckl"),
        orchestrator=OrchestratorConfig(max_iterations=2),
        exploration=StructureGenConfig(),
        oracle=OracleConfig(),
        validation=ValidationConfig(),
        dft=DFTConfig(pseudopotentials={"Si": "Si.upf"}),
    )


def test_orchestrator_initialization(mock_config: Config) -> None:
    explorer = MagicMock()
    selector = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()
    validator = MagicMock()

    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState:
        orch = Orchestrator(mock_config, explorer, selector, oracle, trainer, validator)
        assert orch.config == mock_config
        MockState.assert_called_once()


def test_orchestrator_run_loop(mock_config: Config) -> None:
    explorer = MagicMock()
    selector = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()
    validator = MagicMock()

    # Setup Explorer
    candidates = [CandidateStructure(structure_path=Path("cand.xyz"))]
    explorer.explore.return_value = candidates

    # Setup Selector
    selector.select.return_value = candidates

    # Setup Oracle
    oracle.compute.return_value = [Path("data.extxyz")]

    # Setup Trainer
    trainer.update_dataset.return_value = Path("updated_data.pckl")
    trainer.train.return_value = Path("output.yace")

    # Setup Validator
    validator.validate.return_value = ValidationResult(passed=True)

    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState:
        mock_state_instance = MockState.return_value
        mock_state_instance.load.return_value = WorkflowState(iteration=0)

        orch = Orchestrator(mock_config, explorer, selector, oracle, trainer, validator)

        with patch("shutil.copy"):
            orch.run()

        # 0 < 2 -> Run -> iteration becomes 1
        # 1 < 2 -> Run -> iteration becomes 2
        # 2 < 2 -> Stop.
        assert trainer.train.call_count == 2
        assert mock_state_instance.save.call_count >= 2
        assert explorer.explore.call_count == 2
        assert selector.select.call_count == 2
        assert oracle.compute.call_count == 2
        assert validator.validate.call_count == 2
