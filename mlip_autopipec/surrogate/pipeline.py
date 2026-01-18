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
    Orchestrates the Surrogate Explorer module workflow.

    This class ties together Pre-screening (MACE), Featurization (SOAP),
    and Selection (FPS) to filter and select candidate structures.
    """

    def __init__(self, config: SurrogateConfig):
        """
        Args:
            config: Configuration for the surrogate pipeline.
        """
        self.config = config
        self.mace_client = MaceClient(config)
        self.descriptor_calc = DescriptorCalculator()
        self.sampler = FPSSampler()

    def run(self, candidates: List[Atoms]) -> Tuple[List[Atoms], SelectionResult]:
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
            # 1. Pre-screening
            # We track original indices by attaching them to the Atoms object metadata.
            # This allows us to map back the final selection to the input list.
            candidates_with_meta = self._tag_original_indices(candidates)

            logger.debug("Running MACE pre-screening...")
            kept_atoms, rejected_info = self.mace_client.filter_unphysical(candidates_with_meta)

            logger.info(f"Pre-screening complete. Kept {len(kept_atoms)}/{len(candidates)} structures.")

            if rejected_info:
                 logger.info(f"Rejected {len(rejected_info)} structures. Sample reason: {rejected_info[0].reason}")

            if len(kept_atoms) == 0:
                logger.warning("No candidates passed pre-screening.")
                return [], SelectionResult(selected_indices=[], scores=[])

            # 2. Descriptor Calculation
            logger.debug("Calculating descriptors...")
            try:
                descriptors = self.descriptor_calc.compute_soap(kept_atoms)
            except Exception as e:
                raise RuntimeError(f"Descriptor calculation failed: {e}") from e

            # 3. FPS Selection
            n_samples = min(self.config.fps_n_samples, len(kept_atoms))
            logger.info(f"Selecting {n_samples} structures via FPS...")

            if n_samples == 0:
                 return [], SelectionResult(selected_indices=[], scores=[])

            selected_indices_local, scores = self.sampler.select_with_scores(descriptors, n_samples)

            # 4. Result Resolution
            # Map local indices (into kept_atoms) back to original input indices
            selected_structures, original_indices = self._resolve_selection(
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

    def _tag_original_indices(self, atoms_list: List[Atoms]) -> List[Atoms]:
        """Tags each atom with its original index in the list."""
        # Note: This modifies the atoms in place or we clone them?
        # ASE atoms are mutable. Tagging info is usually safe.
        for i, atom in enumerate(atoms_list):
            if 'info' not in dir(atom):
                atom.info = {}
            atom.info['_original_index'] = i
        return atoms_list

    def _resolve_selection(self, pool: List[Atoms], local_indices: List[int]) -> Tuple[List[Atoms], List[int]]:
        """Resolves the selected subset and maps back to original indices."""
        selected_structures = [pool[i] for i in local_indices]
        original_indices = [atom.info.get('_original_index', -1) for atom in selected_structures]
        return selected_structures, original_indices
