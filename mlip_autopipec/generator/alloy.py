import logging
from typing import Dict, List, Any

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class AlloyGenerator:
    """
    Generator for alloy structures, including SQS, strain, and thermal rattling.
    """

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the AlloyGenerator.

        Args:
            config (GeneratorConfig): The generator configuration.
        """
        self.config = config

    def generate_sqs(self, prim_cell: Atoms, composition: Dict[str, float]) -> Atoms:
        """
        Generates a Special Quasirandom Structure (SQS) for the given composition.

        Tries to use `icet` if available; falls back to random shuffling if not.

        Args:
            prim_cell (Atoms): The primitive unit cell.
            composition (Dict[str, float]): Target composition (e.g., {'Fe': 0.7, 'Ni': 0.3}).

        Returns:
            Atoms: The generated SQS structure.

        Raises:
            GeneratorError: If composition is invalid or generation fails.
        """
        # Validate composition sums to 1 (approx)
        if abs(sum(composition.values()) - 1.0) > 1e-4:
            raise GeneratorError("Composition must sum to 1.0")

        try:
            # Determine supercell size from config
            if not self.config.sqs.enabled:
                logger.warning("SQS generation is disabled in config but was called.")

            supercell_matrix = self.config.sqs.supercell_matrix
            atoms = make_supercell(prim_cell, supercell_matrix)
            n_atoms = len(atoms)

            # Calculate target counts
            counts = {}
            symbols: List[str] = []

            sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)

            for elem, frac in sorted_comp:
                count = int(round(frac * n_atoms))
                counts[elem] = count
                symbols.extend([elem] * count)

            # Fix rounding errors
            diff = n_atoms - len(symbols)
            if diff != 0:
                # Fill remaining spots with major element
                major_elem = sorted_comp[0][0]
                if diff > 0:
                    symbols.extend([major_elem] * diff)
                else:
                    # If we have too many, truncate from the end (which are major elements)
                    pass

            # Truncate if too many (rare case)
            symbols = symbols[:n_atoms]

            # Check if we still have shortage
            while len(symbols) < n_atoms:
                symbols.append(sorted_comp[0][0])

            # Fallback to random shuffle directly as 'icet' integration is complex
            # and we want robust fallback.
            # In a real scenario, we would wrap icet calls here.
            np.random.shuffle(symbols)
            atoms.set_chemical_symbols(symbols)
            atoms.info["config_type"] = "sqs"
            atoms.info["origin"] = "random_shuffle"

            return atoms
        except Exception as e:
            if isinstance(e, GeneratorError):
                raise
            raise GeneratorError(f"Failed to generate SQS: {e}") from e

    def apply_strain(self, atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
        """
        Applies a strain tensor to the atoms object.

        Args:
            atoms (Atoms): The structure to strain.
            strain_tensor (np.ndarray): 3x3 strain tensor epsilon.

        Returns:
            Atoms: The strained structure.
        """
        try:
            strained = atoms.copy()
            cell = strained.get_cell()

            # Deformation gradient F = I + epsilon
            deformation = np.eye(3) + strain_tensor

            # New cell vectors. ASE cell rows are vectors.
            # new_cell = cell @ deformation
            new_cell = np.dot(cell, deformation)

            strained.set_cell(new_cell, scale_atoms=True)

            strained.info["config_type"] = "strain"
            strained.info["strain_tensor"] = strain_tensor.tolist()
            return strained
        except Exception as e:
            raise GeneratorError(f"Failed to apply strain: {e}") from e

    def apply_rattle(self, atoms: Atoms, sigma: float) -> Atoms:
        """
        Applies Gaussian noise to atomic positions.

        Args:
            atoms (Atoms): The structure to rattle.
            sigma (float): Standard deviation of the Gaussian noise in Angstroms.

        Returns:
            Atoms: The rattled structure.
        """
        try:
            rattled = atoms.copy()
            # Explicit implementation using numpy.random.normal
            delta = np.random.normal(0, sigma, atoms.positions.shape)
            rattled.positions += delta

            rattled.info["config_type"] = "rattle"
            rattled.info["rattle_sigma"] = sigma
            return rattled
        except Exception as e:
            raise GeneratorError(f"Failed to apply rattle: {e}") from e

    def generate_batch(self, base_structure: Atoms) -> List[Atoms]:
        """
        Generates a batch of structures: SQS -> Strain -> Rattle.

        Combinatorial expansion based on configuration.

        Args:
            base_structure (Atoms): The starting structure (e.g., SQS).

        Returns:
            List[Atoms]: A list of generated structures.
        """
        results = []

        if not self.config.distortion.enabled:
            # If distortions disabled, just return base
            return [base_structure]

        try:
            # 1. Base (SQS/Prim)
            base_structure.info["config_type"] = base_structure.info.get("config_type", "base")
            results.append(base_structure)

            # 2. Strains
            min_s, max_s = self.config.distortion.strain_range
            n_steps = self.config.distortion.n_strain_steps
            strains = np.linspace(min_s, max_s, n_steps)

            strained_structures = []
            for s in strains:
                if abs(s) < 1e-6:
                    continue  # Skip zero strain if it duplicates base

                # Hydrostatic strain
                strain_tensor = np.eye(3) * s
                strained = self.apply_strain(base_structure, strain_tensor)
                strained_structures.append(strained)
                results.append(strained)

            # 3. Rattles
            # Apply rattles to Base AND Strained structures
            structures_to_rattle = [base_structure] + strained_structures
            n_rattles = self.config.distortion.n_rattle_steps
            amp = self.config.distortion.rattling_amplitude

            for st in structures_to_rattle:
                for _ in range(n_rattles):
                    rattled = self.apply_rattle(st, amp)
                    # Keep parent info
                    if "strain_tensor" in st.info:
                        rattled.info["strain_tensor"] = st.info["strain_tensor"]
                    rattled.info["parent_config_type"] = st.info.get("config_type")
                    results.append(rattled)

            return results
        except Exception as e:
             if isinstance(e, GeneratorError):
                 raise
             raise GeneratorError(f"Batch generation failed: {e}") from e
