import logging
import sys

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from mlip_autopipec.config.schemas.generator import SQSConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class SQSStrategy:
    """
    Strategy for generating Special Quasirandom Structures (SQS).
    """

    def __init__(self, config: SQSConfig) -> None:
        self.config = config

    def generate(self, prim_cell: Atoms, composition: dict[str, float]) -> Atoms:
        """
        Generates an SQS supercell.

        Args:
            prim_cell: Primitive cell.
            composition: Target composition dictionary.

        Returns:
            Atoms: Generated SQS structure.
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
            comp_dict = composition
            sorted_comp = sorted(comp_dict.items(), key=lambda x: x[1], reverse=True)

            symbols: list[str] = []
            current_count = 0

            for i, (elem, frac) in enumerate(sorted_comp):
                if i == len(sorted_comp) - 1:
                     # Last element gets the rest
                     count = n_atoms - current_count
                else:
                     count = int(round(frac * n_atoms))

                symbols.extend([elem] * count)
                current_count += count

            # Ensure exact length (though logic above should cover it)
            if len(symbols) != n_atoms:
                # Should not happen with logic above, but safety check
                logger.warning(f"SQS symbol count mismatch: {len(symbols)} vs {n_atoms}. Truncating/Filling.")
                symbols = symbols[:n_atoms]
                while len(symbols) < n_atoms:
                    symbols.append(sorted_comp[0][0])

            # 3. Try icet if available
            if "icet" in sys.modules:
                # Placeholder for icet logic
                pass

            np.random.shuffle(symbols)
            atoms.set_chemical_symbols(symbols)
            atoms.info["config_type"] = "sqs"
            atoms.info["origin"] = "random_shuffle"

        except Exception as e:
            msg = f"Failed to generate SQS: {e}"
            raise GeneratorError(msg) from e

        return atoms
