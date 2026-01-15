# ruff: noqa: D101
"""Handles the screening of structures using a surrogate model."""

import logging
from typing import Any, List

import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

from mlip_autopipec.config_schemas import SurrogateModelParams

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SurrogateModelScreener:
    """Screens a list of atomic structures using a surrogate MLIP model."""

    def __init__(self, config: SurrogateModelParams):
        """Initialise the SurrogateModelScreener.

        Args:
            config: The Pydantic model containing surrogate model parameters.

        """
        self.config = config
        self._calculator = self._load_model()

    def _load_model(self) -> Any:
        """Load the surrogate model calculator.

        Returns:
            An ASE calculator object for the surrogate model.

        """
        logger.info(f"Loading surrogate model from: {self.config.model_path}")
        try:
            return mace_mp(
                model=self.config.model_path, device="cpu", default_dtype="float64"
            )
        except FileNotFoundError:
            logger.exception(
                f"Surrogate model file not found at: {self.config.model_path}"
            )
            raise
        except Exception as e:
            logger.exception(
                "An unexpected error occurred while loading the MACE model."
            )
            raise RuntimeError("Failed to load surrogate model.") from e

    def screen(self, candidates: List[Atoms]) -> List[Atoms]:
        """Filter candidates based on predicted energy from the surrogate model.

        Calculates the potential energy for each candidate structure. Structures
        with an energy per atom above the configured threshold, or those that
        produce invalid energy values (NaN, Inf), are discarded.

        Args:
            candidates: A list of ASE Atoms objects to be screened.

        Returns:
            A list of ASE Atoms objects that passed the screening.

        """
        if not candidates:
            return []

        screened_list = []
        for i, atoms in enumerate(candidates):
            try:
                atoms.calc = self._calculator
                energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                energy_per_atom = energy / len(atoms)

                if np.isnan(energy) or np.isinf(energy):
                    logger.warning(
                        f"Structure {i} yielded an invalid energy value (NaN or Inf). "
                        "Discarding."
                    )
                    continue

                if energy_per_atom < self.config.energy_threshold_ev:
                    screened_list.append(atoms)
                else:
                    logger.debug(
                        f"Discarding structure with energy "
                        f"{energy_per_atom:.2f} eV/atom (threshold: "
                        f"{self.config.energy_threshold_ev:.2f} eV/atom)."
                    )
            except Exception:
                logger.exception(
                    f"An error occurred during energy calculation for structure {i}. "
                    "Discarding."
                )
        return screened_list
