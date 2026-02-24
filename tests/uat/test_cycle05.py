"""UAT for Cycle 05 (MACE Distillation)."""

from unittest.mock import MagicMock

import pytest

from pyacemaker.core.config import (
    DFTConfig,
    DistillationConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
)
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def uat_config(tmp_path):
    config = MagicMock(spec=PYACEMAKERConfig)
    config.version = "0.1.0"
    config.project = ProjectConfig(name="uat_cycle05", root_dir=tmp_path)
    config.distillation = DistillationConfig()
    config.distillation.enable_mace_distillation = True
    config.oracle = OracleConfig(
        dft=DFTConfig(pseudopotentials={"Fe": "dummy.upf"}),
        mock=True
    )
    # Ensure config structure for mace batch size exists
    config.oracle.mace = MagicMock()
    config.oracle.mace.batch_size = 50
    return config

def test_full_distillation_workflow(uat_config, tmp_path):
    """Verify the complete 7-step MACE distillation workflow."""

    # Mock dependencies
    dataset_manager = MagicMock()
    structure_generator = MagicMock()
    oracle = MagicMock()
    mace_trainer = MagicMock()
    mace_oracle = MagicMock()
    pacemaker_trainer = MagicMock()
    active_learner = MagicMock()

    # Setup mocks for artifacts
    active_learner.run_loop.return_value = tmp_path / "fine_tuned.model"
    pacemaker_trainer.train.return_value = tmp_path / "pace.yace"
    mace_trainer.train.return_value = MagicMock(path=tmp_path / "mace_final.model")

    # Instantiate Orchestrator with explicit dependencies
    orchestrator = Orchestrator(
        config=uat_config,
        base_dir=tmp_path,
        structure_generator=structure_generator,
        oracle=oracle,
        mace_trainer=mace_trainer,
        mace_oracle=mace_oracle,
        pacemaker_trainer=pacemaker_trainer,
        active_learner=active_learner
    )

    # Run
    result = orchestrator.run()

    assert result.status == "success"
    assert "potential" in result.artifacts

    # Verify state transitions
    state_file = tmp_path / "pipeline_state.json"
    assert state_file.exists()
    state = PipelineState.model_validate_json(state_file.read_text())
    assert state.current_step == 8
    assert state.completed_steps == [1, 2, 3, 4, 5, 6, 7]

    # Verify step executions
    dataset_manager.save_metadata_stream.assert_called() # Called in step 1, 4, 5
    active_learner.run_loop.assert_called_once() # Step 2
    pacemaker_trainer.train.assert_called() # Step 6, 7
