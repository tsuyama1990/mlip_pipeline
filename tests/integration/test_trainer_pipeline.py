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
            mock=False,
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
    # Note: PacemakerWrapper is instantiated inside PacemakerTrainer's __init__,
    # so we must patch the class where it's used BEFORE initialization if possible,
    # or mock the instance on the trainer object.
    # Since Orchestrator is already initialized, we need to access its trainer.

    # Check if trainer is indeed PacemakerTrainer (it should be default)
    # But wait, ModuleFactory created it.

    # Better approach: Patch subprocess.run used by wrapper, OR patch wrapper method on the instance.
    # Since we want to verify arguments passed to wrapper, patching the wrapper instance is best.

    # Orchestrator -> Trainer (PacemakerTrainer) -> wrapper (PacemakerWrapper)

    from pyacemaker.trainer.pacemaker import PacemakerTrainer
    if isinstance(orchestrator.trainer, PacemakerTrainer):
        with (
            patch.object(orchestrator.trainer.wrapper, 'train_from_input') as mock_train,
            patch.object(orchestrator.trainer, '_generate_input_yaml', wraps=orchestrator.trainer._generate_input_yaml) as mock_gen_yaml
        ):
            mock_pot_path = project_dir / "trained.yace"
            mock_pot_path.touch()
            mock_train.return_value = mock_pot_path

            # 5. Execute Training Phase
            orchestrator._run_training_phase()

            # 6. Verifications

            # A. Verify potential was stored
            assert orchestrator.current_potential is not None
            assert orchestrator.current_potential.type == "PACE"
            # Filename is randomized by PacemakerTrainer
            assert orchestrator.current_potential.path.name.startswith("pace_model_")
            assert orchestrator.current_potential.path.name.endswith(".yace")

            # B. Verify Trainer Wrapper was called
            mock_train.assert_called_once()

            # C. Verify arguments passed to input generation
            # We check if the config dict passed to _generate_input_yaml matches expectation
            mock_gen_yaml.assert_called_once()
            gen_args = mock_gen_yaml.call_args
            config_dict = gen_args[0][0]
            assert config_dict["cutoff"] == 4.5

            # D. Verify arguments passed to wrapper
            call_args = mock_train.call_args
            input_yaml_arg = call_args[0][0]
            assert Path(input_yaml_arg).name == "input.yaml"

        # D. Verify dataset splitter processed items
        # 5 items total
        assert orchestrator.processed_items_count == 5
