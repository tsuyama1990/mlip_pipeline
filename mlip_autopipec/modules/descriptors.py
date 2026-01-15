# ruff: noqa: D101
"""Handles the calculation of structural descriptors."""

import logging
from typing import List

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from mlip_autopipec.config_schemas import SOAPParams

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SOAPDescriptorCalculator:
    """Calculates SOAP descriptors for a list of atomic structures.

    This class encapsulates the configuration and execution of descriptor
    calculations using the dscribe library, separating the data processing
    logic from the main workflow orchestration.
    """

    def __init__(self, soap_params: SOAPParams, species: List[str]):
        """Initialise the SOAPDescriptorCalculator.

        Args:
            soap_params: The Pydantic model containing SOAP hyperparameters.
            species: A list of chemical symbols present in the structures.

        """
        self.soap_params = soap_params
        self.species = species
        self._soap_generator = self._configure_generator()

    def _configure_generator(self) -> SOAP:
        """Configure the dscribe SOAP generator based on the input parameters.

        Returns:
            An instance of the dscribe SOAP generator.

        """
        logger.info(
            f"Configuring SOAP generator for species: {self.species} with "
            f"r_cut={self.soap_params.r_cut}, n_max={self.soap_params.n_max}, "
            f"l_max={self.soap_params.l_max}."
        )
        return SOAP(
            species=self.species,
            r_cut=self.soap_params.r_cut,
            n_max=self.soap_params.n_max,
            l_max=self.soap_params.l_max,
            sigma=self.soap_params.atomic_sigma,
            periodic=True,
            sparse=False,
            average="outer",
        )

    def calculate(self, structures: List[Atoms]) -> np.ndarray:
        """Calculate the SOAP descriptors for a list of structures.

        Args:
            structures: A list of ASE Atoms objects.

        Returns:
            A 2D NumPy array where each row is the average SOAP descriptor
            vector for a structure.

        """
        if not structures:
            return np.array([])

        logger.info(f"Calculating SOAP descriptors for {len(structures)} structures.")
        try:
            descriptors = self._soap_generator.create(structures, n_jobs=-1)
            return descriptors  # type: ignore[no-any-return]
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during SOAP descriptor calculation."
            )
            raise RuntimeError("Failed to calculate SOAP descriptors.") from e
