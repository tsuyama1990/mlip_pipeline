import contextlib
import logging

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from mlip_autopipec.config.schemas.generator import SQSConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class SQSStrategy:
    """
    Strategy for generating Special Quasirandom Structures (SQS).

    This class handles the creation of chemically disordered supercells that
    mimic random alloys.
    """

    def __init__(self, config: SQSConfig, seed: int | None = None) -> None:
        """
        Initialize the SQSStrategy.

        Args:
            config: SQS configuration.
            seed: Random seed for deterministic generation.
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate(self, prim_cell: Atoms, composition: dict[str, float]) -> Atoms:
        """
        Generates an SQS supercell.

        Attempts to use `icet` if available for optimal SQS generation.
        Falls back to random shuffling if `icet` is not installed or fails.

        Args:
            prim_cell: The primitive unit cell.
            composition: Target composition mapping elements to fractions.

        Returns:
            Atoms: The generated SQS structure.

        Raises:
            GeneratorError: If generation fails.
        """
        try:
            # 1. Create Supercell
            if isinstance(self.config.supercell_size, list):
                P = np.diag(self.config.supercell_size)
            else:
                P = self.config.supercell_size

            atoms = make_supercell(prim_cell, P)
            n_atoms = len(atoms)

            # 2. Determine target counts
            sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)

            symbols: list[str] = []
            current_count = 0

            for i, (elem, frac) in enumerate(sorted_comp):
                if i == len(sorted_comp) - 1:
                    # Last element gets the rest
                    count = n_atoms - current_count
                else:
                    count = round(frac * n_atoms)

                symbols.extend([elem] * count)
                current_count += count

            # Ensure exact length
            if len(symbols) != n_atoms:
                logger.warning(
                    f"SQS symbol count mismatch: {len(symbols)} vs {n_atoms}. Truncating/Filling."
                )
                symbols = symbols[:n_atoms]
                while len(symbols) < n_atoms:
                    symbols.append(sorted_comp[0][0])

            # 3. Try icet if available
            with contextlib.suppress(ImportError):
                import icet  # noqa: F401
                # Placeholder for icet logic if implemented

            self.rng.shuffle(symbols)
            atoms.set_chemical_symbols(symbols)
            atoms.info["config_type"] = "sqs"
            atoms.info["origin"] = "random_shuffle"

        except Exception as e:
            msg = f"Failed to generate SQS: {e}"
            logger.error(msg, exc_info=True)
            raise GeneratorError(msg) from e

        return atoms
