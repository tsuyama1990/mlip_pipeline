import itertools
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.inference.runner import LammpsRunner
from mlip_autopipec.orchestration.interfaces import BuilderProtocol, SurrogateProtocol
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper
from mlip_autopipec.utils.embedding import EmbeddingExtractor

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.workflow import WorkflowManager

logger = logging.getLogger(__name__)

T = TypeVar("T")

def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

class PhaseExecutor:
    """
    Handles the execution of individual workflow phases.
    Decouples logic from WorkflowManager and allows for dependency injection.
    """

    def __init__(self, manager: "WorkflowManager") -> None:
        self.manager = manager
        self.config = manager.config
        self.db = manager.db_manager
        self.queue = manager.task_queue

        # Lazy dependencies
        self._builder: BuilderProtocol | None = manager.builder
        self._surrogate: SurrogateProtocol | None = manager.surrogate

    def _create_qe_runner(self) -> QERunner:
        if not self.config.dft_config:
            raise ValueError("DFT configuration is missing.")
        return QERunner(self.config.dft_config)

    def _create_lammps_runner(self, work_dir: Path) -> LammpsRunner:
        if not self.config.inference_config:
            raise ValueError("Inference configuration is missing.")
        return LammpsRunner(self.config.inference_config, work_dir)

    def _create_pacemaker_wrapper(self, config: "TrainingConfig", work_dir: Path) -> PacemakerWrapper:
        return PacemakerWrapper(config, work_dir)

    def execute_exploration(self) -> None:
        """Execute Phase A: Exploration."""
        logger.info("Phase A: Exploration")
        try:
            if not self._builder:
                self._builder = StructureBuilder(self.config)

            batch_size = 100
            total_generated = 0

            # Chunked processing
            for candidate_batch in chunked(self._builder.build(), batch_size):
                if self.config.surrogate_config:
                    if not self._surrogate:
                        self._surrogate = SurrogatePipeline(self.config.surrogate_config)

                    selected, _ = self._surrogate.run(candidate_batch)
                    logger.info(f"Batch: Generated {len(candidate_batch)}, Selected {len(selected)}")
                else:
                    selected = candidate_batch

                for atoms in selected:
                    self.db.save_candidate(
                        atoms,
                        {"status": "pending", "generation": self.manager.state.current_generation},
                    )
                total_generated += len(selected)

            logger.info(f"Exploration complete. Total candidates saved: {total_generated}")

        except Exception:
            logger.exception("Exploration phase failed")

    def execute_dft(self) -> None:
        """Execute Phase B: DFT Labeling."""
        logger.info("Phase B: DFT Labeling")
        try:
            batch_size = 50
            pending_entries = self.db.select_entries("status=pending")

            total_success = 0
            processed_count = 0

            if self.config.dft_config:
                runner = self._create_qe_runner()

                for batch in chunked(pending_entries, batch_size):
                    if not batch:
                        continue

                    atoms_list = [at for _, at in batch]
                    ids = [i for i, _ in batch]

                    logger.info(f"Submitting DFT batch of {len(atoms_list)} structures.")

                    futures = self.queue.submit_dft_batch(runner.run, atoms_list)
                    results = self.queue.wait_for_completion(futures)

                    for atoms, db_id, res in zip(atoms_list, ids, results, strict=True):
                        if res:
                            try:
                                self.db.save_dft_result(
                                    atoms,
                                    res,
                                    {
                                        "status": "training",
                                        "generation": self.manager.state.current_generation,
                                    },
                                )
                                self.db.update_status(db_id, "labeled")
                                total_success += 1
                            except Exception:
                                logger.exception(f"Failed to save DFT result for ID {db_id}")
                        else:
                            logger.warning(f"DFT failed for atom ID {db_id}.")

                    processed_count += len(batch)

            logger.info(f"DFT Phase complete. Processed: {processed_count}, Success: {total_success}")

        except Exception:
            logger.exception("DFT phase failed")

    def execute_training(self) -> None:
        """Execute Phase C: Training."""
        logger.info("Phase C: Training")
        try:
            if not self.config.training_config:
                logger.warning("No Training Config. Skipping training.")
                return

            dataset_builder = DatasetBuilder(self.db)
            # Use default if template_file is None
            # template_path = getattr(self.config.training_config, "template_file", None) or Path("input.yaml")

            # Using self.manager.work_dir
            logger.info("Exporting training data...")
            dataset_builder.export(self.config.training_config, self.manager.work_dir)

            logger.info("Initializing Pacemaker...")
            wrapper = self._create_pacemaker_wrapper(self.config.training_config, self.manager.work_dir)

            # Assuming previous generation's potential can be used as initial
            initial_potential = None
            prev_gen = self.manager.state.current_generation - 1
            if prev_gen >= 0:
                 prev_pot = self.manager.work_dir / "potentials" / f"generation_{prev_gen}.yace"
                 if prev_pot.exists():
                     initial_potential = prev_pot

            logger.info(f"Starting training (Gen {self.manager.state.current_generation})...")
            result = wrapper.train(initial_potential=initial_potential)

            if result.success and result.potential_path:
                logger.info(f"Training complete. Potential at: {result.potential_path}")
                # Save potential to generation specific path
                pot_dir = self.manager.work_dir / "potentials"
                pot_dir.mkdir(exist_ok=True)
                dest = pot_dir / f"generation_{self.manager.state.current_generation}.yace"

                # Copy or move
                try:
                    import shutil
                    shutil.copy2(result.potential_path, dest)
                    # Also update 'current.yace' link/copy
                    current = self.manager.work_dir / "current.yace"
                    shutil.copy2(result.potential_path, current)
                except Exception:
                    logger.exception("Failed to save potential artifacts")
            else:
                logger.error("Training failed.")

        except Exception:
            logger.exception("Training phase failed")

    def execute_inference(self) -> bool:
        """
        Execute Phase D: Inference.

        Returns:
            True if new candidates were extracted (active learning triggered), False otherwise.
        """
        logger.info("Phase D: Inference")
        try:
            if not self.config.inference_config:
                logger.warning("No Inference Config. Skipping inference.")
                return False

            # 1. Locate Potential
            potential_path = self.manager.work_dir / "current.yace"

            if not potential_path.exists():
                logger.error(f"Potential file not found at {potential_path}")
                return False

            # 2. Select Structure for MD
            # Pick the last completed structure efficiently
            # Using generator to avoid loading all into memory
            last_atom_gen = self.db.select(selection="status=training", sort="-id", limit=1)
            start_atoms = next(last_atom_gen, None)

            if not start_atoms:
                logger.warning("No structures available to start MD.")
                return False

            # 3. Run Inference
            runner = self._create_lammps_runner(self.manager.work_dir / "inference")
            result = runner.run(start_atoms, potential_path)

            # 4. Active Learning Logic
            if result.uncertain_structures:
                logger.info("High uncertainty detected. Extracting candidates...")
                embedding_config = EmbeddingConfig()
                extractor = EmbeddingExtractor(embedding_config)

                from ase.io import read

                for dump_path in result.uncertain_structures:
                    try:
                        frames = read(dump_path, index=":", format="lammps-dump-text")

                        if not frames:
                            continue

                        frame = frames[-1]

                        # Simplified: Re-assign symbols based on types if available
                        species = sorted(set(start_atoms.get_chemical_symbols()))
                        if 'type' in frame.arrays:
                            types = frame.arrays['type']
                            symbols = [species[t-1] for t in types]
                            frame.set_chemical_symbols(symbols)

                        if 'c_gamma' in frame.arrays:
                            gammas = frame.arrays['c_gamma']
                            max_idx = int(gammas.argmax())

                            extracted_atoms = extractor.extract(frame, max_idx)

                            self.db.save_candidate(
                                extracted_atoms,
                                {
                                    "status": "pending",
                                    "generation": self.manager.state.current_generation,
                                    "config_type": "active_learning"
                                }
                            )
                            return True

                    except Exception:
                        logger.exception(f"Failed to process dump file {dump_path}")

            logger.info("Inference finished without high uncertainty.")
            return False

        except Exception:
            logger.exception("Inference phase failed")
            return False
