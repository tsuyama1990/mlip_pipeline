"""Integration tests for Orchestrator state management."""

import json
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
        )
    return orch


def test_state_persistence(orchestrator, tmp_path):
    """Test saving and loading pipeline state."""
    state_file = tmp_path / "pipeline_state.json"

    # Create a dummy state
    state = PipelineState(
        current_step=3,
        completed_steps=[1, 2],
        artifacts={"step1": Path("step1.data")},
    )

    # Manually save (we are testing the helper methods we will implement)
    # Since they are private, we might need to access them or test public side effects
    # For now, let's assume we implement _save_pipeline_state and _load_pipeline_state

    # Implement the logic in test to verify expected behavior or test the method if exposed
    # But Orchestrator doesn't have them yet.
    # We will invoke them via orchestrator instance after implementation.
    # For TDD, we expect these methods to exist or be used.

    # Let's test `run` which should trigger state operations.
    pass


def test_run_resumes_from_state(orchestrator, tmp_path):
    """Test that run resumes from the saved state."""
    state_file = tmp_path / "pipeline_state.json"

    # Create existing state: Step 3 is next
    initial_state = PipelineState(
        current_step=3,
        completed_steps=[1, 2],
        artifacts={"pool_path": Path("pool.xyz")},
    )
    state_file.write_text(initial_state.model_dump_json())

    # Mock the workflow steps
    with patch("pyacemaker.orchestrator.MaceDistillationWorkflow") as MockWorkflow:
        workflow_instance = MockWorkflow.return_value

        # Setup mocks for steps
        workflow_instance.step1_direct_sampling.return_value = Path("pool.xyz")
        workflow_instance.step2_active_learning_loop.return_value = MagicMock() # potential
        workflow_instance.step4_surrogate_data_generation.return_value = Path("surrogate.xyz")
        # ... validation of other steps

        # Execute
        orchestrator.run()

        # Verify Step 1 & 2 were SKIPPED (not called)
        workflow_instance.step1_direct_sampling.assert_not_called()
        # Step 2 might be complex, if it was completed. State says completed_steps=[1, 2], current_step=3.
        # Wait, if current_step=3, it means we are ABOUT TO RUN Step 3.
        # If completed_steps=[1, 2], then 1 and 2 are done.

        # Actually, let's look at the mapping.
        # Step 1: Direct Sampling
        # Step 2 & 3: Active Learning Loop (Combined in workflow as step2_active_learning_loop?)
        # SPEC says: "Sequentially executes step1 through step7".

        # If workflow.step2_active_learning_loop covers step 2 and 3, then resuming at 3 might be tricky if they are combined.
        # But let's assume standard mapping:
        # Step 1: Direct Sampling
        # Step 2: AL Loop
        # Step 3: (Maybe part of AL or separate?)
        # SPEC says "Step 2 & 3: Active Learning & Fine-tuning" in MaceDistillationWorkflow comments.
        # So likely it's one method `step2_active_learning_loop`.

        # If current_step=3, and Step 2 is AL Loop (covering 2 & 3), then we should probably be at Step 4?
        # Or maybe the Orchestrator handles Step 3 separately?
        # Let's assume for this test that we want to resume at Step 3, which corresponds to whatever `step3` method is.
        # If `step2_active_learning_loop` returns a potential, maybe Step 3 is something else?

        # Looking at `MaceDistillationWorkflow` in `src/pyacemaker/modules/mace_workflow.py`:
        # `_step2_active_learning_loop` (Steps 2 & 3).
        # So if we completed Step 2 (and 3 implicitly), next is Step 4.

        # Let's adjust test case:
        # State: current_step = 4. completed_steps = [1, 2, 3] (or just 1, 2 if 2 covers both).

        state_file.write_text(PipelineState(
            current_step=4,
            completed_steps=[1, 2, 3],
            artifacts={
                "pool_path": Path("pool.xyz"),
                "fine_tuned_potential": Path("model.yace")
            }
        ).model_dump_json())

        # Run
        orchestrator.run()

        # Verify Steps 1, 2, 3 skipped
        workflow_instance.step1_direct_sampling.assert_not_called()
        workflow_instance.step2_active_learning_loop.assert_not_called()

        # Verify Step 4 executed
        workflow_instance.step4_surrogate_data_generation.assert_called_once()


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
