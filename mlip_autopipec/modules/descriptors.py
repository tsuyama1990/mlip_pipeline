"""Handles the calculation of structural descriptors."""

import logging

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from mlip_autopipec.config_schemas import SOAPParams

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SOAPDescriptorCalculator:
    """Calculates SOAP descriptors for a list of atomic structures."""

    def __init__(self, soap_params: SOAPParams, species: list[str]) -> None:
        """Initialise the SOAPDescriptorCalculator."""
        self.soap_params = soap_params
        self.species = species
        self._soap_generator = self._configure_generator()

    def _configure_generator(self) -> SOAP:
        """Configure the dscribe SOAP generator based on the input parameters."""
        logger.info(
            "Configuring SOAP generator for species: %s with r_cut=%s, n_max=%s, l_max=%s.",
            self.species,
            self.soap_params.r_cut,
            self.soap_params.n_max,
            self.soap_params.l_max,
        )
        try:
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
        except (ValueError, TypeError) as e:
            logger.exception("Invalid parameters provided for SOAP configuration.")
            msg = "Failed to configure SOAP generator."
            raise RuntimeError(msg) from e

    def calculate(self, structures: list[Atoms]) -> np.ndarray:
        """Calculate the SOAP descriptors for a list of structures."""
        if not structures:
            return np.array([])

        logger.info("Calculating SOAP descriptors for %s structures.", len(structures))
        try:
            descriptors = self._soap_generator.create(structures, n_jobs=-1)
            return descriptors  # type: ignore[no-any-return]
        except Exception as e:
            logger.exception("An unexpected error occurred during SOAP descriptor calculation.")
            msg = "Failed to calculate SOAP descriptors."
            raise RuntimeError(msg) from e
