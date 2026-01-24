"""Handles the screening of structures using a surrogate model."""

import logging
from typing import Any

import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

from mlip_autopipec.config.models import SurrogateModelParams

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SurrogateModelScreener:
    """Screens a list of atomic structures using a surrogate MLIP model."""

    def __init__(self, config: SurrogateModelParams) -> None:
        """Initialise the SurrogateModelScreener."""
        self.config = config
        self._calculator = self._load_model()

    def _load_model(self) -> Any:
        """Load the surrogate model calculator."""
        logger.info("Loading surrogate model from: %s", self.config.model_path)
        try:
            return mace_mp(model=self.config.model_path, device="cpu", default_dtype="float64")
        except FileNotFoundError:
            logger.exception("Surrogate model file not found at: %s", self.config.model_path)
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred while loading the MACE model.")
            msg = "Failed to load surrogate model."
            raise RuntimeError(msg) from e

    def screen(self, candidates: list[Atoms]) -> list[Atoms]:
        """Filter candidates based on predicted energy from the surrogate model."""
        if not candidates:
            return []

        screened_list = []
        for i, atoms in enumerate(candidates):
            try:
                atoms.calc = self._calculator
                energy = atoms.get_potential_energy()
                energy_per_atom = energy / len(atoms)

                if np.isnan(energy) or np.isinf(energy):
                    logger.warning(
                        "Structure %d yielded an invalid energy value (NaN or Inf). Discarding.", i
                    )
                    continue

                if energy_per_atom < self.config.energy_threshold_ev:
                    screened_list.append(atoms)
                else:
                    logger.debug(
                        "Discarding structure with energy %.2f eV/atom (threshold: %.2f eV/atom).",
                        energy_per_atom,
                        self.config.energy_threshold_ev,
                    )
            except Exception:
                logger.exception(
                    "An error occurred during energy calculation for structure %d. Discarding.", i
                )
        return screened_list
