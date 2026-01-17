"""The Surrogate Explorer module for intelligent candidate selection."""

import logging

import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist

from mlip_autopipec.config.models import ExplorerParams
from mlip_autopipec.modules.descriptors import SOAPDescriptorCalculator
from mlip_autopipec.modules.screening import SurrogateModelScreener

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SurrogateExplorer:
    """
    Orchestrates the intelligent selection of structures.

    This class manages the pipeline for selecting the most diverse and
    stable structures from a candidate pool using surrogate model screening
    and Farthest Point Sampling (FPS).
    """

    def __init__(
        self,
        config: ExplorerParams,
        descriptor_calculator: SOAPDescriptorCalculator,
        screener: SurrogateModelScreener,
    ) -> None:
        """Initialise the SurrogateExplorer with system configuration."""
        self.config = config
        self.descriptor_calculator = descriptor_calculator
        self.screener = screener

    def select(self, candidates: list[Atoms]) -> list[Atoms]:
        """Execute the full surrogate screening and FPS selection pipeline."""
        if not candidates:
            logger.warning("Candidate list is empty. Returning an empty list.")
            return []

        logger.info("Starting selection process with %s candidates.", len(candidates))

        # Stage 1: Screen with surrogate model
        screened_candidates = self.screener.screen(candidates)
        logger.info("After surrogate screening, %s candidates remain.", len(screened_candidates))

        if not screened_candidates:
            logger.warning(
                "No candidates remained after surrogate screening. Returning empty list."
            )
            return []

        num_structures_to_select = self.config.fps.num_structures_to_select
        if len(screened_candidates) <= num_structures_to_select:
            logger.warning(
                "Number of available candidates (%d) is less than or equal to the "
                "requested number (%d). Returning all available candidates.",
                len(screened_candidates),
                num_structures_to_select,
            )
            return screened_candidates

        # Stage 2: Calculate descriptors
        descriptors = self.descriptor_calculator.calculate(screened_candidates)

        # Stage 3: Farthest Point Sampling
        selected_indices = self._farthest_point_sampling(descriptors, num_structures_to_select)

        final_selection = [screened_candidates[i] for i in selected_indices]
        logger.info("Selected %s final candidates via FPS.", len(final_selection))

        return final_selection

    def _farthest_point_sampling(
        self, descriptors: np.ndarray, num_structures_to_select: int
    ) -> list[int]:
        """Select a diverse subset using the Farthest Point Sampling algorithm."""
        if num_structures_to_select >= len(descriptors):
            return list(range(len(descriptors)))

        selected_indices = []
        initial_index = np.random.randint(0, len(descriptors))
        selected_indices.append(int(initial_index))

        min_distances = cdist(descriptors, descriptors[selected_indices, :]).min(axis=1)

        for _ in range(num_structures_to_select - 1):
            next_index = int(np.argmax(min_distances))
            selected_indices.append(next_index)
            new_distances = cdist(descriptors, descriptors[selected_indices[-1:], :]).flatten()
            min_distances = np.minimum(min_distances, new_distances)

        return selected_indices
