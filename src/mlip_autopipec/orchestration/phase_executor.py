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
from mlip_autopipec.orchestration.strategies import GammaSelectionStrategy
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
                        {"status": "pending", "generation": self.manager.state.cycle_index},
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
                                        "generation": self.manager.state.cycle_index,
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
            prev_gen = self.manager.state.cycle_index - 1
            if prev_gen >= 0:
                 prev_pot = self.manager.work_dir / "potentials" / f"generation_{prev_gen}.yace"
                 if prev_pot.exists():
                     initial_potential = prev_pot

            logger.info(f"Starting training (Gen {self.manager.state.cycle_index})...")
            result = wrapper.train(initial_potential=initial_potential)

            if result.success and result.potential_path:
                logger.info(f"Training complete. Potential at: {result.potential_path}")
                # Save potential to generation specific path
                pot_dir = self.manager.work_dir / "potentials"
                pot_dir.mkdir(exist_ok=True)
                dest = pot_dir / f"generation_{self.manager.state.cycle_index}.yace"

                # Update state
                self.manager.state.latest_potential_path = dest

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
        Execute Phase: Exploration (MD Inference).

        Returns:
            True if high uncertainty was detected (halted), False otherwise.
        """
        logger.info("Phase: Inference / Exploration")
        try:
            if not self.config.inference_config:
                logger.warning("No Inference Config. Skipping inference.")
                return False

            # 1. Locate Potential
            potential_path = self.manager.state.latest_potential_path or (self.manager.work_dir / "current.yace")

            if not potential_path.exists():
                logger.error(f"Potential file not found at {potential_path}")
                return False

            # 2. Select Structure for MD
            last_atom_gen = self.db.select(selection="status=training", sort="-id", limit=1)
            start_atoms = next(last_atom_gen, None)

            if not start_atoms:
                logger.warning("No structures available to start MD.")
                return False

            # 3. Run Inference
            runner = self._create_lammps_runner(self.manager.work_dir / "inference")
            result = runner.run(start_atoms, potential_path)

            # 4. Check for Halt
            if result.uncertain_structures:
                logger.info("High uncertainty detected. Extracting raw candidates...")
                embedding_config = EmbeddingConfig()
                extractor = EmbeddingExtractor(embedding_config)

                from ase.io import read

                extracted_count = 0
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

                            # Save as 'screening' status for Selection phase
                            self.db.save_candidate(
                                extracted_atoms,
                                {
                                    "status": "screening",
                                    "generation": self.manager.state.cycle_index,
                                    "config_type": "active_learning"
                                }
                            )
                            extracted_count += 1

                    except Exception:
                        logger.exception(f"Failed to process dump file {dump_path}")

                return extracted_count > 0

            logger.info("Inference finished without high uncertainty.")
            return False

        except Exception:
            logger.exception("Inference phase failed")
            return False

    def execute_selection(self) -> None:
        """
        Execute Phase: Selection.
        Selects from 'screening' candidates and promotes them to 'pending'.
        """
        logger.info("Phase: Selection")
        try:
            # 1. Load candidates pending screening
            screening_entries = list(self.db.select_entries("status=screening"))
            if not screening_entries:
                logger.info("No candidates in screening.")
                return

            candidates = [atoms for _, atoms in screening_entries]
            ids = [i for i, _ in screening_entries]

            logger.info(f"Screening {len(candidates)} candidates.")

            # 2. Initialize Strategy
            potential_path = self.manager.state.latest_potential_path or (self.manager.work_dir / "current.yace")

            if not self.config.training_config:
                 logger.warning("No Training Config for Selection Strategy.")
                 # Fallback: select all if no training config (can't run pacemaker)
                 selected_indices = range(len(candidates))
            else:
                 pacemaker = self._create_pacemaker_wrapper(self.config.training_config, self.manager.work_dir)
                 strategy = GammaSelectionStrategy(pacemaker, EmbeddingConfig()) # Using default embedding config

                 # Strategy returns Atoms objects, but we need to map back to DB IDs to update status
                 # This implies strategy should probably take IDs or return indices?
                 # My strategy returns Atoms.

                 # Let's modify usage:
                 # We can just update ALL 'screening' to 'rejected' first, then 'pending' for selected?
                 # Or better: match by some property? Atoms equality is hard.

                 # Refactoring Strategy to return indices might be better?
                 # But sticking to current implementation:

                 # Since GammaSelectionStrategy uses PacemakerWrapper.select_active_set which works on file,
                 # and returns indices relative to the input list.
                 # I can rely on list order preservation.

                 # Let's peek at GammaSelectionStrategy implementation:
                 # it calls self.pacemaker.select_active_set(candidates, potential_path) which returns indices.
                 # and then returns [candidates[i] for i in indices].

                 # I should assume order is preserved.
                 # But I need to invoke pacemaker active set directly to get indices if I want IDs.
                 # Or I can update GammaSelectionStrategy to return indices or (Atoms, ID) tuples?

                 # I'll rely on the fact that `strategy.pacemaker.select_active_set` is what does the work.
                 # I will copy logic here to get indices, effectively bypassing Strategy class if it hides indices.
                 # Or better, I should have designed Strategy to return indices.

                 # Implementation Fix:
                 # I will just select all for now if I can't easily map back.
                 # Wait, I implemented GammaSelectionStrategy.

                 indices = pacemaker.select_active_set(candidates, potential_path)
                 selected_indices = set(indices)

            # 3. Update Status
            selected_count = 0
            for i, db_id in enumerate(ids):
                if i in selected_indices:
                    self.db.update_status(db_id, "pending")
                    selected_count += 1
                else:
                    self.db.update_status(db_id, "rejected")

            logger.info(f"Selection complete. Selected: {selected_count}, Rejected: {len(candidates) - selected_count}")

        except Exception:
            logger.exception("Selection phase failed")
