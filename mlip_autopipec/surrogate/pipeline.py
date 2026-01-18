import logging
from typing import List, Tuple, Optional
from ase import Atoms
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig, SelectionResult
from mlip_autopipec.surrogate.mace_client import MaceClient
from mlip_autopipec.surrogate.descriptors import DescriptorCalculator
from mlip_autopipec.surrogate.sampling import FPSSampler

logger = logging.getLogger(__name__)

class SurrogatePipeline:
    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.mace_client = MaceClient(config)
        self.descriptor_calc = DescriptorCalculator()
        self.sampler = FPSSampler()

    def run(self, candidates: List[Atoms]) -> Tuple[List[Atoms], SelectionResult]:
        """
        Runs the full surrogate pipeline:
        1. Pre-screening (filter high forces)
        2. Descriptor calculation
        3. Diversity selection (FPS)

        Returns:
            - selected_structures: List[Atoms]
            - selection_result: SelectionResult (indices in the ORIGINAL candidates list, scores)
        """
        if not candidates:
            return [], SelectionResult(selected_indices=[], scores=[])

        logger.info(f"Starting surrogate pipeline with {len(candidates)} candidates.")

        # 1. Pre-screening
        # Note: MaceClient returns kept_atoms and rejected info.
        # But to map back to original indices, we need to track indices.
        # It's better if we keep track of original indices throughout.

        # Add original index to atoms.info if not present? Or use auxiliary list.
        # Let's map kept candidates back to original indices.
        # But filter_unphysical returns new list.
        # Let's trust filter_unphysical to be sequential for kept ones?
        # Actually, MaceClient.filter_unphysical returns kept_atoms.
        # We need to know which ones were kept to map indices later if we want original indices.
        # However, filter_unphysical logic iterates and builds list.
        # Let's modify filter_unphysical to return indices of kept ones?
        # Or just do it here if MaceClient exposes predict.

        # Let's use filter_unphysical but we need to know the mapping.
        # Since I can't easily change MaceClient interface without breaking tests I just wrote
        # (though I wrote them, so I can), let's see.
        # The MaceClient.filter_unphysical returns kept atoms.
        # I can just pass the kept atoms to the next stage.
        # But the Requirement says "SelectionResult.selected_indices".
        # Does this mean indices in the filtered set or original set?
        # Usually original set is more useful for tracking.
        # But if we discard exploding ones, they are gone.
        # "ingest 10,000 raw candidates and output the Top 100".

        # Let's assume selected_indices refers to the indices within the *filtered* pool passed to FPS?
        # OR original. The SPEC says "Return filtered list and rejected list".
        # And "FPS ... returns indices of the N most distinct samples".

        # If I return indices relative to the input `candidates`, I need to track them.

        # Let's refine the flow.

        # 1. Predict forces for ALL.
        # (MaceClient.filter_unphysical does this internally).
        # We might want to just call predict and do filtering here to track indices?
        # Or update MaceClient to return indices.
        # Given strict TDD, I should have thought of this.
        # But let's look at `filter_unphysical` implementation again.
        # It returns kept_atoms.

        # I will assume for now we work with the kept list.
        # The indices in SelectionResult will be relative to the *candidates passed to run*?
        # No, that's hard if we filter.
        # If I filter, the list shrinks.

        # Let's do this:
        # We will augment atoms with `_original_index` in `info`.

        for i, atom in enumerate(candidates):
            atom.info['_original_index'] = i

        kept_atoms, rejected_info = self.mace_client.filter_unphysical(candidates)

        logger.info(f"Pre-screening complete. Kept {len(kept_atoms)}/{len(candidates)} structures.")

        if len(kept_atoms) == 0:
            logger.warning("No candidates passed pre-screening.")
            return [], SelectionResult(selected_indices=[], scores=[])

        # 2. Descriptor Calculation
        logger.info("Calculating descriptors...")
        descriptors = self.descriptor_calc.compute_soap(kept_atoms)

        # 3. FPS Selection
        n_samples = min(self.config.fps_n_samples, len(kept_atoms))
        logger.info(f"Selecting {n_samples} structures via FPS...")

        if n_samples == 0:
             return [], SelectionResult(selected_indices=[], scores=[])

        selected_indices_local, scores = self.sampler.select_with_scores(descriptors, n_samples)

        # Map local indices (into kept_atoms) back to original indices if stored
        # But wait, kept_atoms[i] corresponds to descriptors[i].

        selected_structures = [kept_atoms[i] for i in selected_indices_local]

        # Retrieve original indices
        original_indices = [atom.info.get('_original_index', -1) for atom in selected_structures]

        # If for some reason _original_index is missing (shouldn't happen), we return -1 or local?
        # Let's assume it works.

        result = SelectionResult(
            selected_indices=original_indices,
            scores=scores
        )

        return selected_structures, result
