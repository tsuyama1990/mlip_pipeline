import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase.io import write

from pyacemaker.core.base import BaseComponent
from pyacemaker.core.config import DistillationConfig
from pyacemaker.core.dataset import DatasetManager
from pyacemaker.core.utils import (
    stream_metadata_to_atoms,
    validate_structure_integrity_atoms,
)
from pyacemaker.core.validation import validate_safe_path
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.domain_models.structure import StructureMetadata
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.modules.oracle import MaceSurrogateOracle
from pyacemaker.modules.structure_generator import StructureGenerator
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.oracle.manager import OracleManager

if TYPE_CHECKING:
    from ase import Atoms


logger = logging.getLogger(__name__)


class MaceDistillationWorkflow(BaseComponent):
    """
    Workflow manager for the MACE Distillation process (Cycle 05/06).
    Orchestrates the steps:
    1. Direct Sampling (Structure Generation)
    2. Active Learning Loop (MACE <-> DFT)
    3. Final MACE Training
    4. Surrogate Data Generation
    5. Surrogate Labeling
    6. Pacemaker Training
    7. Delta Learning (Optional)
    """

    def __init__(
        self,
        config: DistillationConfig,
        dataset_manager: DatasetManager,
        active_learner: ActiveLearner,
        structure_generator: StructureGenerator,
        oracle: OracleManager,
        mace_oracle: MaceSurrogateOracle,
        pacemaker_trainer: PacemakerTrainer,
        mace_trainer: Any,  # MaceTrainer type
        work_dir: Path,
    ):
        super().__init__(config)
        self.dataset_manager = dataset_manager
        self.active_learner = active_learner
        self.structure_generator = structure_generator
        self.oracle = oracle
        self.mace_oracle = mace_oracle
        self.pacemaker_trainer = pacemaker_trainer
        self.mace_trainer = mace_trainer
        self.work_dir = work_dir

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _get_pool_path(self, step_num: int) -> Path:
        """Generates a safe path for the structure pool."""
        filename = f"step{step_num}_pool.xyz"
        return validate_safe_path(self.work_dir / filename)

    def step1_direct_sampling(self, state: PipelineState) -> PipelineState:
        """
        Step 1: Generate initial pool of candidate structures.
        """
        logger.info("Step 1: Direct Sampling (Structure Generation)")
        pool_path = self._get_pool_path(1)

        # Generate structures as a stream
        candidates_iter = self.structure_generator.generate_candidates()

        # Validation wrapper for the stream
        def validated_stream(iterator: Iterator[StructureMetadata]) -> Iterator[StructureMetadata]:
            for s in iterator:
                # We can add lightweight validation here if needed on metadata
                yield s

        # Save stream to file (memory safe)
        count = self.dataset_manager.save_metadata_stream(
            validated_stream(candidates_iter),
            pool_path
        )

        logger.info(f"Generated {count} candidates in {pool_path}")

        state.current_step = 2
        state.artifacts["pool_path"] = str(pool_path)
        return state

    def step2_active_learning_loop(self, state: PipelineState) -> PipelineState:
        """
        Step 2: Active Learning Loop (MACE <-> DFT).
        Refines the MACE model using DFT data.
        """
        logger.info("Step 2: Active Learning Loop")
        pool_path_str = state.artifacts.get("pool_path")
        if not pool_path_str:
            raise ValueError("pool_path not found in state artifacts")
        pool_path = Path(pool_path_str)

        try:
            # Active Learner manages the loop: select -> label -> train -> repeat
            final_model_path = self.active_learner.run_loop(
                pool_path=pool_path,
                work_dir=self.work_dir
            )

            state.artifacts["mace_model_path"] = str(final_model_path)
            state.current_step = 3
            logger.info(f"Active Learning loop completed. Model: {final_model_path}")

        except Exception as e:
            logger.error(f"Active Learning Loop failed: {e}")
            raise

        return state

    def step3_final_mace_training(self, state: PipelineState) -> PipelineState:
        """
        Step 3: Train final MACE model on all gathered data.
        Usually redundant if AL loop returns the best model, but explicit step for clarity.
        """
        logger.info("Step 3: Final MACE Training")
        # In this workflow, the AL loop's result is often sufficient.
        state.current_step = 4
        return state

    def step4_surrogate_data_generation(self, state: PipelineState) -> PipelineState:
        """
        Step 4: Generate large pool for Pacemaker training (Surrogate Data).
        """
        logger.info("Step 4: Surrogate Data Generation")
        surrogate_pool_path = self._get_pool_path(4)

        # Generate a large number of structures
        # Use islice to limit count if generator is infinite, but ensure it's large enough
        target_count = 10000  # Example target, should be config driven

        candidates_iter = self.structure_generator.generate_candidates()
        limited_iter = islice(candidates_iter, target_count)

        count = self.dataset_manager.save_metadata_stream(limited_iter, surrogate_pool_path)

        logger.info(f"Generated {count} surrogate candidates in {surrogate_pool_path}")
        state.artifacts["surrogate_pool_path"] = str(surrogate_pool_path)
        state.current_step = 5
        return state

    def step5_surrogate_labeling(self, state: PipelineState) -> PipelineState:
        """
        Step 5: Label the surrogate pool using the MACE model.
        """
        logger.info("Step 5: Surrogate Labeling with MACE")

        surrogate_pool_str = state.artifacts.get("surrogate_pool_path")
        if not surrogate_pool_str:
            raise ValueError("surrogate_pool_path not found in artifacts")
        surrogate_pool_path = Path(surrogate_pool_str)

        labeled_pool_path = self.work_dir / "step5_surrogate_labeled.xyz"

        mace_model_str = state.artifacts.get("mace_model_path")
        if not mace_model_str:
             raise ValueError("mace_model_path not found in artifacts")
        mace_model_path = Path(mace_model_str)

        # Update Oracle with the trained model
        self.mace_oracle.update_model(mace_model_path)

        # Stream load -> Label -> Stream save
        # This prevents loading all 10k+ structures into memory

        # 1. Load stream
        input_iter = self.dataset_manager.load_iter(surrogate_pool_path)

        # 2. Convert to Atoms for calculator
        atoms_iter = stream_metadata_to_atoms(input_iter)

        # 3. Label stream (batching handled inside calculator/oracle usually,
        # but here we might need to be careful. ASE calculator is per-atoms usually)

        def labeled_stream_gen(iterator: Iterator["Atoms"]) -> Iterator["Atoms"]:
            batch_size = 100 # Configurable
            batch: list[Atoms] = []
            for atoms in iterator:
                # Validation
                validate_structure_integrity_atoms(atoms)

                batch.append(atoms)
                if len(batch) >= batch_size:
                    # Process batch
                    for a in batch:
                        # Calc energy/forces
                        a.calc = self.mace_oracle.calculator
                        try:
                            a.get_potential_energy()
                            a.get_forces()
                            yield a
                        except Exception as e:
                            logger.warning(f"Failed to label structure: {e}")
                            # Skip bad structures
                            continue
                    batch = []

            # Leftovers
            for a in batch:
                a.calc = self.mace_oracle.calculator
                try:
                    a.get_potential_energy()
                    a.get_forces()
                    yield a
                except Exception as e:
                    logger.warning(f"Failed to label structure: {e}")

        # 4. Save stream
        count = 0
        # Initialize file
        if labeled_pool_path.exists():
            labeled_pool_path.unlink()

        # Write incrementally
        with open(labeled_pool_path, "w") as f:
             for atoms in labeled_stream_gen(atoms_iter):
                 write(f, atoms, format="extxyz")
                 count += 1

        logger.info(f"Labeled {count} surrogate structures in {labeled_pool_path}")
        state.artifacts["labeled_surrogate_path"] = str(labeled_pool_path)
        state.current_step = 6
        return state

    def step6_pacemaker_base_training(self, state: PipelineState) -> PipelineState:
        """
        Step 6: Train Pacemaker potential on the surrogate data.
        """
        logger.info("Step 6: Pacemaker Base Training")
        labeled_data_str = state.artifacts.get("labeled_surrogate_path")
        if not labeled_data_str:
            raise ValueError("labeled_surrogate_path not found in artifacts")
        labeled_data_path = Path(labeled_data_str)

        output_pot_path = self.pacemaker_trainer.train(
            training_data=labeled_data_path,
            test_data=None,
            run_dir=self.work_dir / "pacemaker_run"
        )

        state.artifacts["pacemaker_potential_path"] = str(output_pot_path)
        state.current_step = 7
        return state

    def step7_delta_learning(self, state: PipelineState) -> PipelineState:
        """
        Step 7: Delta Learning (Optional).
        """
        logger.info("Step 7: Delta Learning check")
        # Logic for delta learning if config enabled
        # ...
        state.current_step = 8 # Done
        return state
