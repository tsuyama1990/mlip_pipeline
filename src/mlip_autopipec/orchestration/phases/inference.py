import logging

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from typing import Union
from mlip_autopipec.inference.eon import EONWrapper
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
            runner: Union[LammpsRunner, EONWrapper]
            if self.config.inference_config.active_engine == "eon":
                if not self.config.inference_config.eon:
                    logger.error("EON engine selected but no EON config provided.")
                    return False
                runner = EONWrapper(self.config.inference_config.eon, self.manager.work_dir / "inference_eon")
            else:
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
                        # Differentiate extraction based on engine
                        if self.config.inference_config.active_engine == "eon":
                            # EON produces .con files usually. Use auto-detect or explicit 'eon'
                            frames = [read(dump_path)]
                        else:
                            frames = read(dump_path, index=":", format="lammps-dump-text")

                        if not frames:
                            continue

                        frame = frames[-1]

                        # Simplified: Re-assign symbols based on types if available (LAMMPS specific)
                        if self.config.inference_config.active_engine == "lammps":
                            species = sorted(set(start_atoms.get_chemical_symbols()))
                            if 'type' in frame.arrays:
                                types = frame.arrays['type']
                                symbols = [species[t-1] for t in types]
                                frame.set_chemical_symbols(symbols)

                        # Logic for extraction:
                        # For LAMMPS, we check per-atom gamma (c_gamma).
                        # For EON, the whole structure is 'uncertain' usually (Saddle point high gamma).
                        # The driver checks max_gamma.
                        # If EON returns it, it's likely the whole structure we want to add.

                        if 'c_gamma' in frame.arrays:
                            gammas = frame.arrays['c_gamma']
                            max_idx = int(gammas.argmax())
                            extracted_atoms = extractor.extract(frame, max_idx)
                        else:
                            # For EON, we take the whole frame as candidate if explicit atom-wise gamma is missing
                            # Or we should re-calculate? No, just add the structure.
                            extracted_atoms = frame

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
