"""UAT for Cycle 05: Delta Learning and Orchestration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.trainer.pacemaker import PacemakerTrainer


@pytest.fixture
def uat_config(tmp_path):
    """Create UAT config."""
    config = MagicMock(spec=PYACEMAKERConfig)
    config.project = MagicMock()
    config.project.root_dir = tmp_path
    config.distillation = MagicMock()
    config.distillation.enable_mace_distillation = True
    config.distillation.step7_pacemaker_finetune = MagicMock()
    config.distillation.step7_pacemaker_finetune.enable = True
    config.distillation.step7_pacemaker_finetune.weight_dft = 10.0

    config.trainer = MagicMock()
    config.trainer.mock = True
    config.trainer.model_dump.return_value = {}

    config.orchestrator = MagicMock()
    config.orchestrator.dataset_file = "dataset.pckl"
    config.orchestrator.validation_file = "validation.pckl"

    config.version = "1.0"

    return config


def test_scenario_01_delta_learning(uat_config, tmp_path):
    """
    Scenario 01: Delta Learning Success.
    Verify that the system can fine-tune the base potential with DFT data.
    """
    # Setup Paths
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    dataset_path = data_dir / "dataset.pckl"
    dataset_path.touch()

    # Create base potential
    base_pot_path = models_dir / "base_potential.yace"
    base_pot_path.touch()

    # Initialize Trainer
    # Disable mock mode to ensure wrapper is called (we mock the wrapper itself)
    uat_config.trainer.mock = False
    trainer = PacemakerTrainer(uat_config)

    # Mock wrapper
    with patch.object(trainer.wrapper, 'train_from_input') as mock_train:
        # Create the expected output file
        final_pot_file = models_dir / "final_potential.yace"
        final_pot_file.touch()
        mock_train.return_value = final_pot_file

        # Capture input.yaml content
        captured_input_data = {}

        def train_side_effect(input_file, work_dir, initial_potential=None):
            with open(input_file) as f:
                captured_input_data.update(yaml.safe_load(f))
            return final_pot_file

        mock_train.side_effect = train_side_effect

        # Also mock dataset saving to avoid empty dataset error
        with patch.object(trainer.dataset_manager, 'save_iter'):
            # Dummy generator
            def dummy_gen():
                from uuid import uuid4
                from pyacemaker.domain_models.structure import StructureMetadata
                yield StructureMetadata(id=uuid4(), energy=-10.0, forces=[[0,0,0]])

            def side_effect_save(iterator, *args, **kwargs):
                list(iterator) # Consume

            trainer.dataset_manager.save_iter.side_effect = side_effect_save

            # Execute Delta Learning
            from pyacemaker.domain_models.models import Potential, PotentialType
            base_potential = Potential(path=base_pot_path, type=PotentialType.PACE, version="1.0", metrics={}, parameters={})

            # Patch stream_metadata_to_atoms
            with patch("pyacemaker.trainer.pacemaker.stream_metadata_to_atoms") as mock_stream:
                mock_stream.side_effect = lambda x: x

                final_pot = trainer.train(
                    dummy_gen(),
                    initial_potential=base_potential,
                    weight_dft=10.0
                )

            # Verify Output
            assert final_pot.path.name.endswith(".yace")

            # Verify Inputs
            assert mock_train.called

            # Verify captured input data
            assert captured_input_data["loss"]["w_energy"] == 10.0
            assert captured_input_data["loss"]["w_forces"] == 10.0

            # Verify wrapper call arguments
            args, kwargs = mock_train.call_args
            # train_from_input(input_file, output_dir, initial_potential=None)
            # args[0] is input_file, args[1] is output_dir
            # initial_potential is passed as kwarg or positional depending on call
            # Let's check kwargs first
            passed_initial = kwargs.get('initial_potential')
            if passed_initial is None and len(args) > 2:
                passed_initial = args[2]

            assert passed_initial == base_pot_path


def test_scenario_02_pipeline_idempotency(uat_config, tmp_path):
    """
    Scenario 02: Pipeline Idempotency (Crash Recovery).
    Verify that the system can resume from a failed or interrupted state.
    """
    # Orchestrator with mocked modules
    with patch("pyacemaker.orchestrator.ModuleFactory"):
        orchestrator = Orchestrator(
            config=uat_config,
            structure_generator=MagicMock(),
            oracle=MagicMock(),
            trainer=MagicMock(),
            dynamics_engine=MagicMock(),
            validator=MagicMock(),
            mace_trainer=MagicMock(),
            mace_oracle=MagicMock(),
            active_learner=MagicMock(),
        )

    # 1. Simulate interruption at Step 3
    state_file = tmp_path / "pipeline_state.json"
    state = PipelineState(
        current_step=4, # Ready for Step 4
        completed_steps=[1, 2, 3],
        artifacts={
            "pool_path": Path("pool.xyz"),
            "fine_tuned_potential": Path("model.yace")
        },
    )
    state_file.write_text(state.model_dump_json())

    # 2. Run Orchestrator
    with patch("pyacemaker.orchestrator.MaceDistillationWorkflow") as MockWorkflow:
        workflow = MockWorkflow.return_value

        # Setup mocks
        workflow.step1_direct_sampling.return_value = Path("pool.xyz")
        workflow.step2_active_learning_loop.return_value = MagicMock()
        workflow.step4_surrogate_data_generation.return_value = Path("surrogate.xyz")
        workflow.step5_surrogate_labeling.return_value = Path("surr_ds.xyz")
        workflow.step6_pacemaker_base_training.return_value = MagicMock(path=Path("base.yace"))
        workflow.step7_delta_learning.return_value = MagicMock(path=Path("final.yace"))

        orchestrator.run()

        # Verify Step 1-3 skipped
        workflow.step1_direct_sampling.assert_not_called()
        workflow.step2_active_learning_loop.assert_not_called()

        # Verify Step 4+ executed
        workflow.step4_surrogate_data_generation.assert_called_once()
        workflow.step5_surrogate_labeling.assert_called_once()
        workflow.step6_pacemaker_base_training.assert_called_once()
        workflow.step7_delta_learning.assert_called_once()

        # Verify state was updated to finished
        final_state = PipelineState.model_validate_json(state_file.read_text())
        assert final_state.current_step == 8
        assert 7 in final_state.completed_steps
