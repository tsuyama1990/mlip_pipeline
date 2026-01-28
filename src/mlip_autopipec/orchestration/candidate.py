import logging
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from mlip_autopipec.dft.embedding import ClusterEmbedder
from mlip_autopipec.domain_models.candidate import CandidateConfig
from mlip_autopipec.generator.transformations import apply_rattle
from mlip_autopipec.training.pacemaker import PacemakerWrapper

logger = logging.getLogger(__name__)


class CandidateProcessor:
    """
    Handles processing of halted structures: extraction, perturbation, selection, and embedding.
    """

    def __init__(self, config: CandidateConfig, pacemaker: PacemakerWrapper):
        self.config = config
        self.pacemaker = pacemaker
        self.embedder = ClusterEmbedder(cutoff=config.cluster_cutoff)

    def process(
        self, halted_dump: Path, potential_path: Path, elements: list[str]
    ) -> list[Atoms]:
        """
        Processes a halted dump file to produce DFT-ready candidates.
        """
        logger.info(f"Processing halted dump: {halted_dump}")

        if not halted_dump.exists():
            logger.error(f"Dump file not found: {halted_dump}")
            return []

        # 1. Read Structure
        try:
            # Read last frame
            atoms = read(halted_dump, index=-1, format="lammps-dump-text")
            if not isinstance(atoms, Atoms):
                # Should not happen with index=-1
                atoms = atoms[-1] # type: ignore

            # Remap types to elements
            current_numbers = atoms.get_atomic_numbers()
            if len(elements) == 0:
                 logger.warning("No elements provided for remapping. Keeping types.")
            elif max(current_numbers) > len(elements):
                logger.error(
                    f"Dump contains type {max(current_numbers)} but only {len(elements)} elements provided."
                )
                # Fail safe or return?
                return []
            else:
                symbols = [elements[n - 1] for n in current_numbers]
                atoms.set_chemical_symbols(symbols)

            # 2. Identify Center (Max Gamma)
            center_index = 0
            if "c_gamma" in atoms.arrays:
                gammas = atoms.get_array("c_gamma")
                center_index = int(np.argmax(gammas))
                logger.debug(f"Max gamma found at atom {center_index}: {gammas[center_index]}")
            else:
                logger.warning("No 'c_gamma' found in dump. Using atom 0 as center.")

            # 3. Cut Cluster
            cluster = self.embedder.embed(atoms, center_index=center_index)

            # 4. Generate Perturbations
            candidates = [cluster]
            logger.info(f"Generating {self.config.num_perturbations} perturbations.")
            for _ in range(self.config.num_perturbations):
                try:
                    perturbed = apply_rattle(cluster, sigma=self.config.perturbation_radius)
                    candidates.append(perturbed)
                    logger.info("Perturbation added.")
                except Exception as e:
                    logger.warning(f"Perturbation failed: {e}", exc_info=True)

            # 5. Select Best Candidates
            # Write candidates to temp file
            selected_candidates = []
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                write(str(tmp_path), candidates, format="extxyz")

                # Check if potential exists (it should)
                if potential_path.exists():
                    selected_indices = self.pacemaker.select_active_set(tmp_path, potential_path)

                    for i in selected_indices:
                        if 0 <= i < len(candidates):
                            selected_candidates.append(candidates[i])
                        else:
                            logger.warning(f"Selected index {i} out of bounds (0-{len(candidates)-1})")

                    if not selected_candidates:
                        logger.info("No candidates selected by pace_activeset. Fallback to base cluster.")
                        selected_candidates = [cluster]
                else:
                    logger.warning("Potential path does not exist. Skipping selection, keeping base cluster.")
                    selected_candidates = [cluster]

            except Exception:
                logger.exception("Active set selection failed. Fallback to base cluster.")
                selected_candidates = [cluster]
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            return selected_candidates

        except Exception:
            logger.exception("Failed to process halted dump.")
            return []
