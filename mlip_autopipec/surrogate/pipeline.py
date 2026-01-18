import logging

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SelectionResult, SurrogateConfig
from mlip_autopipec.surrogate.candidate_manager import CandidateManager
from mlip_autopipec.surrogate.descriptors import DescriptorCalculator
from mlip_autopipec.surrogate.mace_client import MaceClient
from mlip_autopipec.surrogate.sampling import FPSSampler

logger = logging.getLogger(__name__)

class SurrogatePipeline:
    """
    Orchestrates the Surrogate Explorer module workflow.

    This class ties together Pre-screening (MACE), Featurization (SOAP),
    and Selection (FPS) to filter and select candidate structures.
    It delegates data management to CandidateManager.
    """

    def __init__(self, config: SurrogateConfig):
        """
        Args:
            config: Configuration for the surrogate pipeline.
        """
        self.config = config
        self.mace_client = MaceClient(config)
        # Use descriptor_config from SurrogateConfig
        self.descriptor_calc = DescriptorCalculator(config.descriptor_config)
        self.sampler = FPSSampler()

    def run(self, candidates: list[Atoms]) -> tuple[list[Atoms], SelectionResult]:
        """
        Executes the surrogate pipeline on a list of candidate structures.

        Args:
            candidates: List of candidate atomic structures (ase.Atoms).

        Returns:
            A tuple of (selected_structures, selection_result).
            selection_result contains indices relative to the input `candidates` list.

        Raises:
            RuntimeError: If any stage of the pipeline fails.
        """
        if not candidates:
            return [], SelectionResult(selected_indices=[], scores=[])

        logger.info(f"Starting surrogate pipeline with {len(candidates)} candidates.")

        try:
            # 1. Pre-processing
            candidates_with_meta = CandidateManager.tag_candidates(candidates)

            # 2. Pre-screening
            kept_atoms = self._execute_prescreening(candidates_with_meta)
            if not kept_atoms:
                return [], SelectionResult(selected_indices=[], scores=[])

            # 3. Descriptor Calculation
            descriptors = self._execute_featurization(kept_atoms)

            # 4. FPS Selection
            selected_indices_local, scores = self._execute_selection(descriptors, len(kept_atoms))

            # 5. Result Resolution
            selected_structures, original_indices = CandidateManager.resolve_selection(
                kept_atoms, selected_indices_local
            )

            result = SelectionResult(
                selected_indices=original_indices,
                scores=scores
            )

            return selected_structures, result

        except Exception as e:
            logger.error(f"Surrogate pipeline execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Surrogate pipeline execution failed: {e}") from e

    def _execute_prescreening(self, candidates: list[Atoms]) -> list[Atoms]:
        """Runs MACE pre-screening."""
        logger.debug("Running MACE pre-screening...")
        kept_atoms, rejected_info = self.mace_client.filter_unphysical(candidates)

        logger.info(f"Pre-screening complete. Kept {len(kept_atoms)}/{len(candidates)} structures.")

        if rejected_info:
             logger.info(f"Rejected {len(rejected_info)} structures. Sample reason: {rejected_info[0].reason}")

        if len(kept_atoms) == 0:
            logger.warning("No candidates passed pre-screening.")

        return kept_atoms

    def _execute_featurization(self, kept_atoms: list[Atoms]) -> np.ndarray:
        """Computes descriptors."""
        logger.debug("Calculating descriptors...")
        try:
            descriptor_result = self.descriptor_calc.compute_soap(kept_atoms)
            return descriptor_result.features
        except Exception as e:
            raise RuntimeError(f"Descriptor calculation failed: {e}") from e

    def _execute_selection(self, descriptors: np.ndarray, pool_size: int) -> tuple[list[int], list[float]]:
        """Runs FPS selection."""
        n_samples = min(self.config.fps_n_samples, pool_size)
        logger.info(f"Selecting {n_samples} structures via FPS...")

        if n_samples == 0:
             return [], []

        return self.sampler.select_with_scores(descriptors, n_samples)
