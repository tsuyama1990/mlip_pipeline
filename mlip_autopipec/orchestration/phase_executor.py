import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.inference.embedding import EmbeddingExtractor
from mlip_autopipec.inference.runner import LammpsRunner
from mlip_autopipec.config.schemas.inference import EmbeddingConfig
from mlip_autopipec.orchestration.interfaces import BuilderProtocol, SurrogateProtocol
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.training.config_gen import TrainConfigGenerator
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper

if TYPE_CHECKING:
    from mlip_autopipec.orchestration.workflow import WorkflowManager

logger = logging.getLogger(__name__)


class PhaseExecutor:
    """
    Handles the execution of individual workflow phases to decouple logic from WorkflowManager.
    """

    def __init__(self, manager: "WorkflowManager") -> None:
        self.manager = manager
        self.config = manager.config
        self.db = manager.db_manager
        self.queue = manager.task_queue

        # Lazy dependencies
        self._builder: BuilderProtocol | None = manager.builder
        self._surrogate: SurrogateProtocol | None = manager.surrogate

    def execute_exploration(self) -> None:
        """Execute Phase A: Exploration."""
        logger.info("Phase A: Exploration")
        try:
            if not self._builder:
                self._builder = StructureBuilder(self.config)

            candidates = self._builder.build()
            logger.info(f"Generated {len(candidates)} raw candidates.")

            if self.config.surrogate_config:
                if not self._surrogate:
                    self._surrogate = SurrogatePipeline(self.config.surrogate_config)

                selected, _ = self._surrogate.run(candidates)
                logger.info(f"Selected {len(selected)} candidates via Surrogate.")
            else:
                selected = candidates
                logger.info("Surrogate skipped (no config). Using all candidates.")

            for atoms in selected:
                self.db.save_candidate(
                    atoms,
                    {"status": "pending", "generation": self.manager.state.current_generation},
                )

        except Exception:
            logger.exception("Exploration phase failed")

    def execute_dft(self) -> None:
        """Execute Phase B: DFT Labeling."""
        logger.info("Phase B: DFT Labeling")
        try:
            # Use get_entries to get ID for updates
            entries = self.db.get_entries("status=pending")
            if not entries:
                logger.warning("No pending atoms found for DFT.")
                return

            logger.info(f"Found {len(entries)} pending atoms for DFT.")
            atoms_list = [at for _, at in entries]
            ids = [i for i, _ in entries]

            if self.config.dft_config:
                runner = QERunner(self.config.dft_config)
                futures = self.queue.submit_dft_batch(runner.run, atoms_list)
                results = self.queue.wait_for_completion(futures)

                success_count = 0
                for atoms, db_id, res in zip(atoms_list, ids, results, strict=True):
                    if res:
                        try:
                            # Save new row with results
                            self.db.save_dft_result(
                                atoms,
                                res,
                                {
                                    "status": "training",
                                    "generation": self.manager.state.current_generation,
                                },
                            )
                            # Mark old row as processed (labeled)
                            self.db.update_status(db_id, "labeled")

                            success_count += 1
                        except Exception as e:
                            logger.error(f"Failed to save DFT result: {e}")
                    else:
                        logger.warning("DFT failed for an atom.")

                logger.info(f"DFT Phase complete. Success: {success_count}/{len(atoms_list)}")

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
            template_path = getattr(self.config.training_config, "template_file", None) or Path("input.yaml")

            config_gen = TrainConfigGenerator(template_path=template_path)
            wrapper = PacemakerWrapper()

            result = wrapper.train(
                self.config.training_config,
                dataset_builder,
                config_gen,
                self.manager.work_dir,
                self.manager.state.current_generation,
            )
            logger.info(f"Training complete. Potential at: {result.potential_path}")

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
            potential_path = self.manager.work_dir / "potentials" / f"generation_{self.manager.state.current_generation}.yace"
            if not potential_path.exists():
                potential_path = self.manager.work_dir / "current.yace"

            if not potential_path.exists():
                logger.error(f"Potential file not found at {potential_path}")
                return False

            # 2. Select Structure for MD
            # Pick a random completed structure
            completed_atoms = self.db.get_atoms("status=training")

            if not completed_atoms:
                logger.warning("No structures available to start MD.")
                return False

            start_atoms = completed_atoms[-1] # Pick last one

            # 3. Run Inference
            runner = LammpsRunner(self.config.inference_config, self.manager.work_dir / "inference")
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
                        species = sorted(list(set(start_atoms.get_chemical_symbols())))
                        if 'type' in frame.arrays:
                            types = frame.arrays['type']
                            symbols = [species[t-1] for t in types]
                            frame.set_chemical_symbols(symbols)

                        if 'c_gamma' in frame.arrays:
                            gammas = frame.arrays['c_gamma']
                            max_idx = int(gammas.argmax())

                            extracted = extractor.extract(frame, max_idx)

                            # Save to DB
                            self.db.save_candidate(
                                extracted.atoms,
                                {
                                    "status": "pending",
                                    "generation": self.manager.state.current_generation,
                                    "config_type": "active_learning"
                                }
                            )
                            return True

                    except Exception as e:
                        logger.error(f"Failed to process dump file {dump_path}: {e}")

            logger.info("Inference finished without high uncertainty.")
            return False

        except Exception:
            logger.exception("Inference phase failed")
            return False
