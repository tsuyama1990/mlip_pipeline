import logging

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.inference.runner import LammpsRunner
from mlip_autopipec.orchestration.phases.base import BasePhase
from mlip_autopipec.utils.embedding import EmbeddingExtractor

logger = logging.getLogger(__name__)

class InferencePhase(BasePhase):
    def execute(self) -> bool:
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
            runner = LammpsRunner(self.config.inference_config, self.manager.work_dir / "inference")
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
