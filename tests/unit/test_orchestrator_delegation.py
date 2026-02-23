"""Tests for Orchestrator delegation to MACE Workflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
    Validator,
)
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def mace_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Fixture for MACE configuration."""
    config_dict = {
        "project": {"name": "test", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "vasp",
                "pseudopotentials": {"Fe": "pot"},
                "command": "run"
            },
            "mace": {"model_path": "medium", "mock": True}
        },
        "distillation": {
            "enable_mace_distillation": True,
            "step1_direct_sampling": {"target_points": 5},
            "step2_active_learning": {"cycles": 1, "n_select": 2},
            "step3_mace_finetune": {"epochs": 1},
            "step4_surrogate_sampling": {"target_points": 5}
        },
        "version": "0.1.0"
    }
    return PYACEMAKERConfig(**config_dict)

def test_orchestrator_delegates_to_workflow(mace_config: PYACEMAKERConfig) -> None:
    """Test that Orchestrator initializes and runs MaceDistillationWorkflow."""

    # Mock dependencies
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_oracle = MagicMock(spec=Oracle)
    # Oracle must be UncertaintyModel for check
    class MockUncertaintyOracle(Oracle, UncertaintyModel):
        def compute_batch(self, s): return iter([])
        def compute_uncertainty(self, s): return iter([])
        def run(self): return None

    mock_oracle = MockUncertaintyOracle(mace_config)

    mock_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_val = MagicMock(spec=Validator)

    # We need to import DynamicsEngine and Validator to mock them for typing?
    # No, MagicMock satisfies Any.

    # Instantiate Orchestrator
    orch = Orchestrator(
        mace_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        validator=mock_val,
        mace_trainer=mock_mace_trainer
    )

    # Patch the workflow class inside orchestrator module
    with patch('pyacemaker.orchestrator.MaceDistillationWorkflow') as MockWorkflow:
        workflow_instance = MockWorkflow.return_value
        workflow_instance.run.return_value = MagicMock(status="success")

        _ = orch.run()

        # Verify
        MockWorkflow.assert_called_once()
        workflow_instance.run.assert_called_once()

        # Verify arguments passed to Workflow constructor
        call_args = MockWorkflow.call_args[1] # kwargs
        assert call_args["config"] == mace_config
        assert call_args["dataset_manager"] == orch.dataset_manager
        assert call_args["dataset_path"] == orch.dataset_path
        assert call_args["oracle"] == mock_oracle
        assert call_args["trainer"] == mock_trainer
        assert call_args["mace_trainer"] == mock_mace_trainer
        assert call_args["dynamics_engine"] == mock_dyn
        assert call_args["structure_generator"] == mock_sg
