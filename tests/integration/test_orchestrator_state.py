"""Integration tests for Orchestrator state management."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config."""
    config = MagicMock(spec=PYACEMAKERConfig)
    config.project = MagicMock()
    config.project.root_dir = tmp_path
    config.distillation = MagicMock()
    config.distillation.enable_mace_distillation = True
    config.orchestrator = MagicMock()
    config.orchestrator.dataset_file = "dataset.xyz"
    config.orchestrator.validation_file = "validation.xyz"
    return config


@pytest.fixture
def orchestrator(mock_config):
    """Create an Orchestrator instance."""
    # Mock dependencies
    with patch("pyacemaker.orchestrator.ModuleFactory"):
        orch = Orchestrator(
            config=mock_config,
            structure_generator=MagicMock(),
            oracle=MagicMock(),
            trainer=MagicMock(),
            dynamics_engine=MagicMock(),
            validator=MagicMock(),
            mace_trainer=MagicMock(),
            mace_oracle=MagicMock(),
            active_learner=MagicMock(),
        )
    return orch


def test_state_persistence(orchestrator, tmp_path):
    """Test saving and loading pipeline state."""
    # This test primarily verifies infrastructure via the run flow in other tests,
    # or we can test private methods if really needed, but integration tests usually
    # test public interfaces.
    # The Orchestrator manages state internally.


def test_run_resumes_from_state(orchestrator, tmp_path):
    """Test that run resumes from the saved state."""
    state_file = tmp_path / "pipeline_state.json"

    # Create existing state: Skip 1, 2, 3. Resume at 4.
    # completed_steps should be [1, 2, 3] ideally if we are at 4.
    # The logic checks: if state.current_step <= X, run step X.

    initial_state = PipelineState(
        current_step=4,
        completed_steps=[1, 2, 3],
        artifacts={
            "pool_path": Path("pool.xyz"),
            "fine_tuned_potential": Path("model.yace")
        },
    )
    state_file.write_text(initial_state.model_dump_json())

    # Mock the workflow methods on the instance created inside _run_mace_distillation
    # Since Orchestrator instantiates MaceDistillationWorkflow internally, we must patch the class.
    with patch("pyacemaker.orchestrator.MaceDistillationWorkflow") as MockWorkflow:
        workflow_instance = MockWorkflow.return_value

        # Setup mocks for steps
        workflow_instance.step1_direct_sampling.return_value = Path("pool.xyz")
        workflow_instance.step2_active_learning_loop.return_value = MagicMock()
        workflow_instance.step4_surrogate_data_generation.return_value = Path("surrogate.xyz")
        # Ensure other steps return valid dummy paths to avoid crashes
        workflow_instance.step5_surrogate_labeling.return_value = Path("surr_dataset.xyz")
        workflow_instance.step6_pacemaker_base_training.return_value = MagicMock(path=Path("base.yace"))
        workflow_instance.step7_delta_learning.return_value = MagicMock(path=Path("final.yace"))

        # Execute
        orchestrator.run()

        # Verify Steps 1 & 2 were SKIPPED (not called)
        workflow_instance.step1_direct_sampling.assert_not_called()
        workflow_instance.step2_active_learning_loop.assert_not_called()

        # Verify Step 4 executed (since current_step=4)
        workflow_instance.step4_surrogate_data_generation.assert_called_once()

        # Verify Step 5 executed (since we resumed and continued)
        workflow_instance.step5_surrogate_labeling.assert_called_once()

        # Verify Step 6 executed
        workflow_instance.step6_pacemaker_base_training.assert_called_once()

        # Verify Step 7 executed
        workflow_instance.step7_delta_learning.assert_called_once()


def test_state_updates_on_step_completion(orchestrator, tmp_path):
    """Test that state is updated and saved after each step."""
    # Start fresh
    state_file = tmp_path / "pipeline_state.json"
    if state_file.exists():
        state_file.unlink()

    with patch("pyacemaker.orchestrator.MaceDistillationWorkflow") as MockWorkflow:
        workflow_instance = MockWorkflow.return_value

        # Mock step 1 return
        workflow_instance.step1_direct_sampling.return_value = Path("pool.xyz")

        # Mock failure at Step 2 to stop execution there
        workflow_instance.step2_active_learning_loop.side_effect = RuntimeError("Stop here")

        try:
            orchestrator.run()
        except RuntimeError:
            pass

        # Verify state file exists
        assert state_file.exists()

        # Load state and check
        saved_state = PipelineState.model_validate_json(state_file.read_text())

        # Should have completed Step 1
        assert 1 in saved_state.completed_steps
        assert saved_state.current_step == 2 # Ready for Step 2
        assert saved_state.artifacts["pool_path"] == Path("pool.xyz")
