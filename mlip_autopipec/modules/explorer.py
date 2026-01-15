# ruff: noqa: D101
"""The Surrogate Explorer module for intelligent candidate selection."""

import logging
from typing import List

import numpy as np
from ase import Atoms
from mace.calculators import mace_mp
from scipy.spatial.distance import cdist

from mlip_autopipec.config_schemas import ExplorerParams
from mlip_autopipec.modules.descriptors import SOAPDescriptorCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SurrogateExplorer:
    """Orchestrates the intelligent selection of structures.

    This class implements the two-stage filtering process described as Module B
    in the system specification. It first screens a large pool of candidate
    structures using a pre-trained surrogate model (MACE) to discard unphysical
    configurations. It then uses Farthest Point Sampling (FPS) on a descriptor
    representation (SOAP) of the remaining structures to select a maximally
    diverse subset for DFT calculations.
    """

    def __init__(
        self,
        config: ExplorerParams,
        descriptor_calculator: SOAPDescriptorCalculator,
    ):
        """Initialise the SurrogateExplorer with system configuration.

        Args:
            config: The ExplorerParams object containing all parameters.
            descriptor_calculator: An initialized descriptor calculator object.

        """
        self.config = config
        self.descriptor_calculator = descriptor_calculator

    def select(self, candidates: List[Atoms]) -> List[Atoms]:
        """Execute the full surrogate screening and FPS selection pipeline.

        Args:
            candidates: A list of ASE Atoms objects to be filtered and selected.

        Returns:
            A smaller, diverse list of ASE Atoms objects.

        """
        if not candidates:
            logger.warning("Candidate list is empty. Returning an empty list.")
            return []

        logger.info(f"Starting selection process with {len(candidates)} candidates.")

        # Stage 1: Screen with surrogate model
        screened_candidates = self._screen_with_surrogate(candidates)
        logger.info(
            f"After surrogate screening, {len(screened_candidates)} candidates remain."
        )

        if not screened_candidates:
            logger.warning(
                "No candidates remained after surrogate screening. "
                "Returning empty list."
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
        selected_indices = self._farthest_point_sampling(
            descriptors, num_structures_to_select
        )

        final_selection = [screened_candidates[i] for i in selected_indices]
        logger.info(f"Selected {len(final_selection)} final candidates via FPS.")

        return final_selection

    def _screen_with_surrogate(self, candidates: List[Atoms]) -> List[Atoms]:
        """Filter candidates based on predicted energy from a surrogate model.

        Loads a MACE model and calculates the potential energy for each
        candidate structure. Structures with an energy per atom above the
        configured threshold are discarded. Also handles model loading errors
        and invalid energy calculations.

        Args:
            candidates: A list of ASE Atoms objects to be screened.

        Returns:
            A list of ASE Atoms objects that passed the screening.

        """
        model_path = self.config.surrogate_model.model_path
        threshold = self.config.surrogate_model.energy_threshold_ev

        logger.info(f"Loading surrogate model from: {model_path}")
        try:
            calculator = mace_mp(
                model=model_path, device="cpu", default_dtype="float64"
            )
        except FileNotFoundError:
            logger.exception(f"Surrogate model file not found at: {model_path}")
            raise
        except Exception as e:
            logger.exception(
                "An unexpected error occurred while loading the MACE model."
            )
            raise RuntimeError("Failed to load surrogate model.") from e

        screened_list = []
        for i, atoms in enumerate(candidates):
            try:
                atoms.calc = calculator
                energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                energy_per_atom = energy / len(atoms)

                if np.isnan(energy) or np.isinf(energy):
                    logger.warning(
                        f"Structure {i} yielded an invalid energy value (NaN or Inf). "
                        "Discarding."
                    )
                    continue

                if energy_per_atom < threshold:
                    screened_list.append(atoms)
                else:
                    logger.debug(
                        f"Discarding structure with energy "
                        f"{energy_per_atom:.2f} eV/atom "
                        f"(threshold: {threshold:.2f} eV/atom)."
                    )
            except Exception:
                logger.exception(
                    f"An error occurred during energy calculation for structure {i}. "
                    "Discarding."
                )
        return screened_list

    def _farthest_point_sampling(
        self, descriptors: np.ndarray, num_structures_to_select: int
    ) -> List[int]:
        """Select a diverse subset using the Farthest Point Sampling algorithm.

        Iteratively selects `num_structures_to_select` points from the
        descriptor matrix that are maximally distant from the set of already
        chosen points. This ensures a diverse sampling of the descriptor space.

        Args:
            descriptors: A 2D NumPy array of descriptor vectors.
            num_structures_to_select: The number of indices to select.

        Returns:
            A list of integer indices corresponding to the selected descriptors.

        """
        if num_structures_to_select >= len(descriptors):
            return list(range(len(descriptors)))

        selected_indices = []
        # For reproducibility, a fixed seed could be used here.
        initial_index = np.random.randint(0, len(descriptors))
        selected_indices.append(int(initial_index))

        # Initialize an array to store the minimum distance from each point to any
        # of the already selected points.
        min_distances = cdist(descriptors, descriptors[selected_indices, :]).min(axis=1)

        # Iteratively select the point farthest from the current set.
        for _ in range(num_structures_to_select - 1):
            # Find the index of the point with the maximum minimum distance.
            next_index = int(np.argmax(min_distances))
            selected_indices.append(next_index)

            # Update the minimum distances array by calculating the distances to the
            # newly added point and taking the element-wise minimum.
            new_distances = cdist(
                descriptors, descriptors[selected_indices[-1:], :]
            ).flatten()
            min_distances = np.minimum(min_distances, new_distances)

        return selected_indices
