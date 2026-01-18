import logging
from typing import List, Tuple, Optional
from ase import Atoms
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig, SelectionResult
from mlip_autopipec.surrogate.mace_client import MaceClient
from mlip_autopipec.surrogate.descriptors import DescriptorCalculator
from mlip_autopipec.surrogate.sampling import FPSSampler

logger = logging.getLogger(__name__)

class SurrogatePipeline:
    """
    Orchestrates the Surrogate Explorer module.
    1. Pre-screening using MACE (Force filtering).
    2. Descriptor calculation (SOAP).
    3. Diversity Selection (FPS).
    """

    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.mace_client = MaceClient(config)
        self.descriptor_calc = DescriptorCalculator()
        self.sampler = FPSSampler()

    def run(self, candidates: List[Atoms]) -> Tuple[List[Atoms], SelectionResult]:
        """
        Runs the full surrogate pipeline.

        Args:
            candidates: List of candidate ASE Atoms structures.

        Returns:
            Tuple containing:
            - List[Atoms]: The selected diverse structures.
            - SelectionResult: Metadata about the selection (indices, scores).
        """
        if not candidates:
            return [], SelectionResult(selected_indices=[], scores=[])

        logger.info(f"Starting surrogate pipeline with {len(candidates)} candidates.")

        try:
            # 1. Pre-screening
            # Tag original indices to track them after filtering
            for i, atom in enumerate(candidates):
                if 'info' not in dir(atom): # Just in case it's not a standard Atoms object
                    atom.info = {}
                atom.info['_original_index'] = i

            logger.debug("Running MACE pre-screening...")
            kept_atoms, rejected_info = self.mace_client.filter_unphysical(candidates)

            logger.info(f"Pre-screening complete. Kept {len(kept_atoms)}/{len(candidates)} structures.")

            if rejected_info:
                 logger.info(f"Rejected {len(rejected_info)} structures due to physical violations.")

            if len(kept_atoms) == 0:
                logger.warning("No candidates passed pre-screening.")
                return [], SelectionResult(selected_indices=[], scores=[])

            # 2. Descriptor Calculation
            logger.debug("Calculating descriptors...")
            descriptors = self.descriptor_calc.compute_soap(kept_atoms)

            # 3. FPS Selection
            n_samples = min(self.config.fps_n_samples, len(kept_atoms))
            logger.info(f"Selecting {n_samples} structures via FPS...")

            if n_samples == 0:
                 return [], SelectionResult(selected_indices=[], scores=[])

            selected_indices_local, scores = self.sampler.select_with_scores(descriptors, n_samples)

            selected_structures = [kept_atoms[i] for i in selected_indices_local]

            # Retrieve original indices
            original_indices = [atom.info.get('_original_index', -1) for atom in selected_structures]

            result = SelectionResult(
                selected_indices=original_indices,
                scores=scores
            )

            return selected_structures, result

        except Exception as e:
            logger.error(f"Surrogate pipeline failed: {e}", exc_info=True)
            # Depending on policy, we might want to raise or return empty.
            # Usually strict failure is better for debugging.
            raise RuntimeError(f"Surrogate pipeline execution failed: {e}") from e
