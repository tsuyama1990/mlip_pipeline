from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import DistillationConfig, ProjectConfig, PYACEMAKERConfig
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def uat_config(tmp_path):
    config = MagicMock(spec=PYACEMAKERConfig)
    config.version = "0.1.0"
    config.project = ProjectConfig(name="uat_cycle02", root_dir=tmp_path)
    config.distillation = DistillationConfig()
    config.distillation.enable_mace_distillation = True
    config.oracle = MagicMock()
    config.oracle.mace.batch_size = 50
    return config

def test_workflow_dependency_injection(uat_config, tmp_path):
    """Verify that dependencies are correctly passed to the workflow."""

    # Mock dependencies
    mace_trainer = MagicMock()
    mace_oracle = MagicMock()
    pacemaker_trainer = MagicMock()

    orchestrator = Orchestrator(
        config=uat_config,
        base_dir=tmp_path,
        mace_trainer=mace_trainer,
        mace_oracle=mace_oracle,
        pacemaker_trainer=pacemaker_trainer
    )

    # Patch the workflow creation
    with patch("pyacemaker.orchestrator.MaceDistillationWorkflow") as MockWorkflow:
        instance = MockWorkflow.return_value
        # Mock step methods
        instance.step1_direct_sampling.return_value = "pool.xyz"
        instance.step2_active_learning_loop.return_value = MagicMock(path="model.model")

        orchestrator._run_mace_distillation()

        # Verify call args
        MockWorkflow.assert_called_once()
        call_kwargs = MockWorkflow.call_args.kwargs
        assert call_kwargs["mace_trainer"] == mace_trainer
        assert call_kwargs["mace_oracle"] == mace_oracle
        assert call_kwargs["pacemaker_trainer"] == pacemaker_trainer
        assert call_kwargs["batch_size"] == 50

def test_workflow_delegation_flow(uat_config, tmp_path):
    """Verify that orchestrator calls workflow steps in order."""
    orchestrator = Orchestrator(
        config=uat_config,
        base_dir=tmp_path,
        mace_trainer=MagicMock(),
        mace_oracle=MagicMock(),
        pacemaker_trainer=MagicMock()
    )

    with patch("pyacemaker.orchestrator.MaceDistillationWorkflow") as MockWorkflow:
        wf = MockWorkflow.return_value
        # Mock artifacts return
        wf.step1_direct_sampling.return_value = "pool.xyz"
        wf.step2_active_learning_loop.return_value = MagicMock(path="model.model")
        wf.step4_surrogate_data_generation.return_value = "surrogate.xyz"
        wf.step5_surrogate_labeling.return_value = "labeled.xyz"
        wf.step6_pacemaker_base_training.return_value = MagicMock(path="pace.yace")
        wf.step7_delta_learning.return_value = MagicMock(path="final.yace")

        orchestrator.run()

        # Verify steps called
        wf.step1_direct_sampling.assert_called_once()
        wf.step2_active_learning_loop.assert_called_once()
        # step3 is implicit/pass-through in current impl, check if called
        wf.step3_final_mace_training.assert_called_once()
        wf.step4_surrogate_data_generation.assert_called_once()
        wf.step5_surrogate_labeling.assert_called_once()
        wf.step6_pacemaker_base_training.assert_called_once()
        wf.step7_delta_learning.assert_called_once()
