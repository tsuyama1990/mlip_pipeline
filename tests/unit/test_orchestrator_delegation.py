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
from pyacemaker.domain_models.models import Potential, PotentialType
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
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

def test_orchestrator_delegates_to_workflow_steps(mace_config: PYACEMAKERConfig) -> None:
    """Test that Orchestrator initializes and runs MaceDistillationWorkflow steps sequentially."""

    # Mock dependencies
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_oracle = MagicMock(spec=Oracle)
    class MockUncertaintyOracle(Oracle, UncertaintyModel):
        def compute_batch(self, s): return iter([])
        def compute_uncertainty(self, s): return iter([])
        def run(self): return None

    mock_oracle = MockUncertaintyOracle(mace_config)
    mock_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_dyn = MagicMock(spec=DynamicsEngine)
    mock_val = MagicMock(spec=Validator)

    # Instantiate Orchestrator
    orch = Orchestrator(
        mace_config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        validator=mock_val,
        mace_trainer=mock_mace_trainer,
        mace_oracle=mock_oracle
    )

    # Mock return values for steps
    mock_pool_path = Path("pool.pckl")
    mock_fine_tuned_pot = Potential(
        path=Path("ft.model"), type=PotentialType.MACE, version="1.0.0", metrics={}, parameters={}
    )
    mock_surrogate_structures_path = Path("surr_str.pckl")
    mock_surrogate_dataset_path = Path("surr_ds.pckl")
    mock_base_ace_pot = Potential(
        path=Path("base.yace"), type=PotentialType.PACE, version="1.0.0", metrics={}, parameters={}
    )
    mock_final_pot = Potential(
        path=Path("final.yace"), type=PotentialType.PACE, version="1.0.0", metrics={}, parameters={}
    )

    # Patch the workflow class inside orchestrator module
    with patch('pyacemaker.orchestrator.MaceDistillationWorkflow') as MockWorkflow:
        workflow_instance = MockWorkflow.return_value

        # Setup step return values
        workflow_instance.step1_direct_sampling.return_value = mock_pool_path
        workflow_instance.step2_active_learning_loop.return_value = mock_fine_tuned_pot
        workflow_instance.step4_surrogate_data_generation.return_value = mock_surrogate_structures_path
        workflow_instance.step5_surrogate_labeling.return_value = mock_surrogate_dataset_path
        workflow_instance.step6_pacemaker_base_training.return_value = mock_base_ace_pot
        workflow_instance.step7_delta_learning.return_value = mock_final_pot

        result = orch.run()

        assert result.status == "success"

        # Verify calls to steps
        workflow_instance.step1_direct_sampling.assert_called_once()
        workflow_instance.step2_active_learning_loop.assert_called_once()
        workflow_instance.step4_surrogate_data_generation.assert_called_once()
        workflow_instance.step5_surrogate_labeling.assert_called_once()
        workflow_instance.step6_pacemaker_base_training.assert_called_once()
        workflow_instance.step7_delta_learning.assert_called_once()

        # Check call arguments for a specific step, e.g., step4
        workflow_instance.step4_surrogate_data_generation.assert_called_with(
            mace_config.distillation, mock_fine_tuned_pot
        )

        # Verify arguments passed to Workflow constructor
        call_args = MockWorkflow.call_args[1] # kwargs
        assert call_args["config"] == mace_config
        assert call_args["dataset_manager"] == orch.dataset_manager


def test_orchestrator_dependency_injection_for_workflow(mace_config: PYACEMAKERConfig) -> None:
    """Test that we can mock _create_mace_workflow to inject a mock workflow."""

    orch = Orchestrator(mace_config)

    mock_workflow = MagicMock(spec=MaceDistillationWorkflow)
    # Mock return values to prevent crashes
    mock_workflow.step1_direct_sampling.return_value = Path("p")
    mock_workflow.step2_active_learning_loop.return_value = Potential(
        path=Path("p"), type=PotentialType.MACE, version="1.0.0", metrics={}, parameters={}
    )
    mock_workflow.step4_surrogate_data_generation.return_value = Path("p")
    mock_workflow.step5_surrogate_labeling.return_value = Path("p")
    mock_workflow.step6_pacemaker_base_training.return_value = Potential(
        path=Path("p"), type=PotentialType.PACE, version="1.0.0", metrics={}, parameters={}
    )
    mock_workflow.step7_delta_learning.return_value = Potential(
        path=Path("p"), type=PotentialType.PACE, version="1.0.0", metrics={}, parameters={}
    )

    # Patch _create_mace_workflow method on the instance
    # Use setattr to bypass mypy method assignment check
    orch._create_mace_workflow = MagicMock(return_value=mock_workflow)

    orch.run()

    orch._create_mace_workflow.assert_called_once()
    mock_workflow.step1_direct_sampling.assert_called_once()
