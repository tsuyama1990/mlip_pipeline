import logging
from pathlib import Path

from mlip_autopipec.inference.eon import EONWrapper
from mlip_autopipec.inference.processing import CandidateProcessor
from mlip_autopipec.inference.runner import LammpsRunner
from mlip_autopipec.orchestration.phases.base import BasePhase

logger = logging.getLogger(__name__)

# Constants
DEFAULT_POTENTIAL_FILENAME = "current.yace"
INFERENCE_DIR_NAME = "inference"
INFERENCE_EON_DIR_NAME = "inference_eon"

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
            potential_path = self.manager.state.latest_potential_path or (self.manager.work_dir / DEFAULT_POTENTIAL_FILENAME)

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
            runner: LammpsRunner | EONWrapper
            work_dir: Path

            if self.config.inference_config.active_engine == "eon":
                if not self.config.inference_config.eon:
                    logger.error("EON engine selected but no EON config provided.")
                    return False
                work_dir = self.manager.work_dir / INFERENCE_EON_DIR_NAME
                runner = EONWrapper(self.config.inference_config.eon, work_dir)
            else:
                work_dir = self.manager.work_dir / INFERENCE_DIR_NAME
                runner = LammpsRunner(self.config.inference_config, work_dir)

            result = runner.run(start_atoms, potential_path)

            # 4. Check for Halt
            if result.uncertain_structures:
                logger.info("High uncertainty detected. Extracting raw candidates...")

                processor = CandidateProcessor(self.config.inference_config)
                candidates = processor.extract_candidates(result.uncertain_structures, start_atoms)

                # Enrich metadata with generation
                final_candidates = []
                for atoms, meta in candidates:
                    meta["generation"] = self.manager.state.cycle_index
                    final_candidates.append((atoms, meta))

                if final_candidates:
                    self.db.save_candidates(final_candidates)
                    return True

            logger.info("Inference finished without high uncertainty.")
            return False

        except Exception:
            logger.exception("Inference phase failed")
            return False
