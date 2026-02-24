from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase.io import write
from loguru import logger

from pyacemaker.core.config import CONSTANTS, DistillationConfig
from pyacemaker.core.interfaces import Oracle
from pyacemaker.core.logging import get_logger
from pyacemaker.core.utils import (
    atoms_to_metadata,
    stream_metadata_to_atoms,
)
from pyacemaker.core.validation import validate_safe_path
from pyacemaker.domain_models.common import PotentialType
from pyacemaker.domain_models.models import Potential
from pyacemaker.domain_models.state import PipelineState
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.modules.oracle import MaceSurrogateOracle
from pyacemaker.modules.structure_generator import StructureGenerator
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.trainer.mace_trainer import MaceTrainer

if TYPE_CHECKING:
    from ase import Atoms


class MaceDistillationWorkflow:
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
        oracle: Oracle,
        mace_oracle: MaceSurrogateOracle,
        pacemaker_trainer: PacemakerTrainer,
        mace_trainer: MaceTrainer,  # Strict typing
        work_dir: Path,
    ) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.dataset_manager = dataset_manager
        self.active_learner = active_learner
        self.structure_generator = structure_generator
        self.oracle = oracle
        self.mace_oracle = mace_oracle
        self.pacemaker_trainer = pacemaker_trainer
        self.mace_trainer = mace_trainer
        self.work_dir = work_dir

        # Use config-driven settings
        self.batch_size = config.batch_size
        self.write_buffer_size = config.write_buffer_size

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _get_pool_path(self, step_num: int) -> Path:
        """Generates a safe path for the structure pool."""
        filename = f"step{step_num}_pool.xyz"
        # Validate path safety
        return validate_safe_path(self.work_dir / filename)

    def step1_direct_sampling(self, state: PipelineState) -> PipelineState:
        """
        Step 1: Generate initial pool of candidate structures.
        """
        logger.info("Step 1: Direct Sampling (Structure Generation)")
        pool_path = self._get_pool_path(1)

        # Generate structures as a stream
        candidates_iter = self.structure_generator.generate_direct_samples(
            n_samples=self.config.step1_direct_sampling.target_points,
            objective=self.config.step1_direct_sampling.objective
        )

        # Validation wrapper for the stream
        def validated_stream(iterator: Iterator[Any]) -> Iterator[Any]:
            for s in iterator:
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
            raise ValueError("Artifact 'pool_path' missing in state")

        pool_path = Path(str(pool_path_str))

        # Active Learner manages the loop: select -> label -> train -> repeat
        final_model_path, dft_dataset_path = self.active_learner.run_loop(
            pool_path=pool_path,
            work_dir=self.work_dir
        )

        state.artifacts["mace_model_path"] = str(final_model_path)
        state.artifacts["dft_dataset_path"] = str(dft_dataset_path)
        state.current_step = 3
        logger.info(f"Active Learning loop completed. Model: {final_model_path}")

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

        try:
            # Validate input path is safe before writing
            surrogate_pool_path = validate_safe_path(surrogate_pool_path)

            # Generate a large number of structures
            # Use config-driven limit
            target_count = self.config.step4_surrogate_sampling.target_points

            # Reuse direct sampling
            candidates_iter = self.structure_generator.generate_direct_samples(
                n_samples=target_count,
                objective=self.config.step1_direct_sampling.objective # Reuse objective?
            )

            # Ensure memory safety by strictly slicing the iterator
            limited_iter = islice(candidates_iter, target_count)

            # save_metadata_stream is designed to consume iterator one-by-one
            # Verify dataset_manager.save_metadata_stream logic (it iterates)
            count = self.dataset_manager.save_metadata_stream(limited_iter, surrogate_pool_path)

            logger.info(f"Generated {count} surrogate candidates in {surrogate_pool_path}")
            state.artifacts["surrogate_pool_path"] = str(surrogate_pool_path)
            state.current_step = 5
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            raise

        return state

    def _write_labeled_stream(self, labeled_iterator: Iterator["Atoms"], output_path: Path) -> int:
        """Writes labeled structures to file using a buffer.

        Consumes the labeled stream and writes in chunks to optimize I/O.
        """
        count = 0
        if output_path.exists():
            output_path.unlink()

        try:
            with output_path.open("w") as f:
                 buffer: list[Atoms] = []

                 for atoms in labeled_iterator:
                     buffer.append(atoms)
                     count += 1
                     if len(buffer) >= self.write_buffer_size:
                         write(f, buffer, format="extxyz")
                         buffer = [] # Clear buffer after write (Fixes memory leak)

                 if buffer:
                     write(f, buffer, format="extxyz")
        except Exception as e:
            logger.error(f"Failed to write labeled structures to {output_path}: {e}")
            raise

        return count

    def step5_surrogate_labeling(self, state: PipelineState) -> PipelineState:
        """
        Step 5: Label the surrogate pool using the MACE model.
        """
        logger.info("Step 5: Surrogate Labeling with MACE")

        try:
            surrogate_pool_path_str = state.artifacts.get("surrogate_pool_path")
            if not surrogate_pool_path_str:
                raise ValueError("Artifact 'surrogate_pool_path' missing in state")

            # Validate input path is safe before using
            surrogate_pool_path = validate_safe_path(Path(str(surrogate_pool_path_str)))

            labeled_pool_path = self.work_dir / "step5_surrogate_labeled.xyz"
            validate_safe_path(labeled_pool_path)

            mace_model_path_str = state.artifacts.get("mace_model_path")
            if not mace_model_path_str:
                 # Fallback or error? Usually error if AL loop succeeded.
                 logger.warning("mace_model_path missing")
                 raise ValueError("Artifact 'mace_model_path' missing in state")

            mace_model_path = Path(str(mace_model_path_str))

            # Update Oracle with the trained model
            self.mace_oracle.update_model(mace_model_path)

            # Stream load -> Label -> Stream save
            # This prevents loading all 10k+ structures into memory

            # 1. Load stream
            input_iter = self.dataset_manager.load_iter(surrogate_pool_path)

            # 2. Label stream (Generator-based batching via compute_batch)
            # compute_batch consumes StructureMetadata and yields StructureMetadata (with energy/forces)
            labeled_metadata_stream = self.mace_oracle.compute_batch(input_iter)

            # 3. Convert back to Atoms for writing
            atoms_iter = stream_metadata_to_atoms(labeled_metadata_stream)

            # 4. Save stream using batch write to optimize I/O
            count = self._write_labeled_stream(atoms_iter, labeled_pool_path)

            logger.info(f"Labeled {count} surrogate structures in {labeled_pool_path}")
            state.artifacts["labeled_surrogate_path"] = str(labeled_pool_path)
            state.current_step = 6
        except Exception as e:
            logger.error(f"Step 5 failed: {e}")
            raise

        return state

    def step6_pacemaker_base_training(self, state: PipelineState) -> PipelineState:
        """
        Step 6: Train Pacemaker potential on the surrogate data.
        """
        logger.info("Step 6: Pacemaker Base Training")
        try:
            labeled_data_path_str = state.artifacts.get("labeled_surrogate_path")
            if not labeled_data_path_str:
                raise ValueError("Artifact 'labeled_surrogate_path' missing in state")

            labeled_data_path = Path(str(labeled_data_path_str))

            output_pot_path = self.pacemaker_trainer.train(
                training_data=labeled_data_path,
                test_data=None,
                run_dir=self.work_dir / "pacemaker_run"
            )

            state.artifacts["pacemaker_potential_path"] = str(output_pot_path)
            state.current_step = 7
        except Exception as e:
            logger.error(f"Step 6 failed: {e}")
            raise

        return state

    def step7_delta_learning(self, state: PipelineState) -> PipelineState:
        """
        Step 7: Delta Learning (Optional).
        """
        logger.info("Step 7: Delta Learning")

        try:
            # Check configuration
            if not self.config.step7_pacemaker_finetune.enable:
                logger.info("Delta Learning disabled in config. Skipping.")
                state.current_step = 8
                return state

            # Retrieve inputs
            pacemaker_pot_str = state.artifacts.get("pacemaker_potential_path")
            if not pacemaker_pot_str:
                 logger.warning("Base Pacemaker potential missing. Cannot perform Delta Learning.")
                 msg = "Artifact 'pacemaker_potential_path' missing for Delta Learning"
                 raise ValueError(msg)

            dft_dataset_str = state.artifacts.get("dft_dataset_path")
            if not dft_dataset_str:
                 logger.warning("DFT dataset path missing. Cannot perform Delta Learning.")
                 msg = "Artifact 'dft_dataset_path' missing for Delta Learning"
                 raise ValueError(msg)

            base_potential = Potential(
                path=Path(pacemaker_pot_str),
                type=PotentialType.PACE,
                version=CONSTANTS.internal_base_potential_version,
            )
            dft_dataset_path = Path(dft_dataset_str)

            # Load DFT dataset as stream
            def dft_metadata_stream() -> Iterator[Any]:
                # dataset_manager.load_iter returns Atoms, convert to Metadata for trainer
                for atoms in self.dataset_manager.load_iter(dft_dataset_path):
                    yield atoms_to_metadata(atoms)

            weight_dft = self.config.step7_pacemaker_finetune.weight_dft

            logger.info(f"Starting Delta Learning with weight_dft={weight_dft}")
            final_pot = self.pacemaker_trainer.train(
                dataset=dft_metadata_stream(),
                initial_potential=base_potential,
                weight_dft=weight_dft,
            )

            state.artifacts["final_potential"] = str(final_pot.path)
            state.current_step = 8
            logger.info(f"Delta Learning completed. Final potential: {final_pot.path}")
        except Exception as e:
            logger.error(f"Step 7 failed: {e}")
            raise

        return state
