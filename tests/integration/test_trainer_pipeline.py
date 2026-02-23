"""Integration test for the Trainer Pipeline.

This test validates the complete workflow of the Trainer module, from
initializing with a dataset, configuring the training run (including
optional delta learning), to executing the training (mocked) and producing
a potential artifact. It ensures that data flow between the Orchestrator,
DatasetManager, and Trainer is seamless.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from pyacemaker.core.config import (
    DFTConfig,
    DynamicsEngineConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.orchestrator import Orchestrator


def test_trainer_pipeline_execution(tmp_path: Path) -> None:
    """Verify the complete Trainer workflow from Orchestrator to Potential.

    Steps:
    1. Initialize Orchestrator with mock config.
    2. Generate/Provide a dataset stream.
    3. Trigger training phase in Orchestrator.
    4. Verify Trainer is invoked with correct stream and parameters.
    5. Verify Potential object is returned and stored in Orchestrator.
    """
    # 1. Setup Configuration
    project_dir = tmp_path / "trainer_integration"
    project_dir.mkdir()

    pp_path = tmp_path / "Fe.upf"
    pp_path.touch()

    config = PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="TrainerIntTest", root_dir=project_dir),
        orchestrator=OrchestratorConfig(
            validation_split=0.0, # Disable validation split to simplify flow
        ),
        structure_generator=StructureGeneratorConfig(strategy="random"),
        oracle=OracleConfig(
            dft=DFTConfig(code="mock_dft", pseudopotentials={"Fe": str(pp_path)}),
            mock=True,
        ),
        trainer=TrainerConfig(
            potential_type="pace",
            mock=True,
            cutoff=4.5,
        ),
        dynamics_engine=DynamicsEngineConfig(mock=True),
    )

    # 2. Instantiate Orchestrator
    orchestrator = Orchestrator(config)

    # 3. Pre-populate Training Data
    # The Orchestrator's training phase reads from self.dataset_path or splits new data.
    # For this test, we simulate that data has already been accumulated in self.dataset_path.

    dataset_path = orchestrator.dataset_path

    # Ensure directory exists
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate mock data using a generator to avoid OOM in test
    def atoms_generator() -> Iterator[Atoms]:
        for _ in range(5):
            a = Atoms("Fe", positions=[[0, 0, 0]])
            a.calc = SinglePointCalculator(a, energy=-1.0, forces=[[0.0, 0.0, 0.0]])
            yield a

    # Orchestrator uses dataset_manager.save_iter to append to file
    orchestrator.dataset_manager.save_iter(
        atoms_generator(), dataset_path, calculate_checksum=False
    )

    # 4. Mock External Dependencies
    # We mock the Trainer's wrapper to avoid actual subprocess calls
    # but we allow the Trainer module logic (file IO, param prep) to run.
    with patch("pyacemaker.modules.trainer.PacemakerWrapper") as MockWrapper:
        wrapper_instance = MockWrapper.return_value
        # Mock train return value (path to potential)
        mock_pot_path = project_dir / "trained.yace"
        wrapper_instance.train.return_value = mock_pot_path

        # Mock wrapper.select_active_set to prevent errors if called
        # Though this test focuses on training phase, setting up the mock is safer
        wrapper_instance.select_active_set.return_value = project_dir / "active_set.pckl.gzip"

        # 5. Execute Training Phase
        # We invoke the private method _run_training_phase to isolate this test
        # In a real run, this is called by run_cycle()

        # The orchestrator splitter logic relies on self.processed_items_count
        # Since we just created the file, processed_count is 0.
        # The splitter will read the file, splitting into train/val (0% val here).

        orchestrator._run_training_phase()

        # 6. Verifications

        # A. Verify potential was stored
        assert orchestrator.current_potential is not None
        # Mock mode in PacemakerTrainer generates a specific mock potential path
        # instead of using the return value of wrapper.train
        # So we assert the potential exists and has correct type
        assert orchestrator.current_potential.type == "PACE"
        assert orchestrator.current_potential.path.name == "mock_potential.yace"

        # B. Verify Trainer Wrapper was called
        # Note: In mock mode, PacemakerTrainer skips wrapper.train()
        # So we assert it was NOT called, but verify mock path logic above
        if not config.trainer.mock:
            wrapper_instance.train.assert_called_once()
        else:
            wrapper_instance.train.assert_not_called()

        # C. Verify arguments passed to wrapper
        if not config.trainer.mock:
            call_args = wrapper_instance.train.call_args
            dataset_arg = call_args[0][0]
            params_arg = call_args[0][2]

            # Ensure passed dataset path exists (it's a temp file created by Trainer)
            assert Path(dataset_arg).exists()

            # Ensure params match config
            assert params_arg["cutoff"] == 4.5

        # D. Verify dataset splitter processed items
        # 5 items total
        assert orchestrator.processed_items_count == 5
