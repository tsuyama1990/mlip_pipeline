import logging

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from mlip_autopipec.config.schemas.common import Composition
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


class AlloyGenerator:
    """
    Generator for alloy structures, including SQS, strain, and thermal rattling.

    This class handles the creation of Special Quasirandom Structures (SQS) to model
    random alloys, and applies physical distortions (strain and rattling) to explore
    the potential energy surface.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """
        Initialize the AlloyGenerator.

        Args:
            config (GeneratorConfig): The generator configuration containing settings for SQS and distortions.
        """
        self.config = config

    def generate_sqs(self, prim_cell: Atoms, composition: Composition) -> Atoms:
        """
        Generates a Special Quasirandom Structure (SQS) for the given composition.

        This method attempts to create a supercell that matches the target composition
        as closely as possible. It currently implements a fallback strategy using
        random shuffling if advanced SQS generation (e.g., via `icet`) is not available.

        Args:
            prim_cell (Atoms): The primitive unit cell to expand.
            composition (Composition): Target composition map wrapped in a Pydantic model.
                                       Values must sum to approximately 1.0.

        Returns:
            Atoms: The generated SQS supercell structure.

        Raises:
            GeneratorError: If generation fails.
        """
        # Composition validation is handled by the Pydantic model on input.
        comp_dict: dict[str, float] = composition.root

        try:
            # Determine supercell size from config
            if not self.config.sqs.enabled:
                logger.warning("SQS generation is disabled in config but was called.")

            supercell_matrix = self.config.sqs.supercell_matrix
            atoms = make_supercell(prim_cell, supercell_matrix)
            n_atoms = len(atoms)

            # Calculate target counts
            symbols: list[str] = []

            sorted_comp = sorted(comp_dict.items(), key=lambda x: x[1], reverse=True)

            for elem, frac in sorted_comp:
                count = round(frac * n_atoms)
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
            msg = f"Failed to generate SQS: {e}"
            raise GeneratorError(msg) from e

    def apply_strain(self, atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
        """
        Applies a generic strain tensor to the atoms object.

        The new cell is calculated as: cell_new = cell_old @ (I + epsilon).
        Atomic positions are scaled accordingly.

        Args:
            atoms (Atoms): The structure to strain.
            strain_tensor (np.ndarray): A 3x3 symmetric strain tensor (epsilon).

        Returns:
            Atoms: The new strained structure with updated cell dimensions.

        Raises:
            GeneratorError: If matrix operations fail.
        """
        try:
            strained = atoms.copy()
            cell = strained.get_cell()

            # Deformation gradient F = I + epsilon
            deformation = np.eye(3) + strain_tensor

            # New cell vectors. ASE cell rows are vectors.
            new_cell = np.dot(cell, deformation)

            strained.set_cell(new_cell, scale_atoms=True)

            strained.info["config_type"] = "strain"
            strained.info["strain_tensor"] = strain_tensor.tolist()
            return strained
        except Exception as e:
            msg = f"Failed to apply strain: {e}"
            raise GeneratorError(msg) from e

    def apply_rattle(self, atoms: Atoms, sigma: float) -> Atoms:
        """
        Applies random thermal displacement (rattling) to atomic positions.

        Displacements are drawn from a Gaussian distribution centered at 0 with
        standard deviation `sigma`.

        Args:
            atoms (Atoms): The structure to rattle.
            sigma (float): Standard deviation of the displacement in Angstroms.

        Returns:
            Atoms: The rattled structure.

        Raises:
            GeneratorError: If the operation fails.
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
            msg = f"Failed to apply rattle: {e}"
            raise GeneratorError(msg) from e

    def generate_batch(self, base_structure: Atoms) -> list[Atoms]:
        """
        Generates a combinatorial batch of structures from a base structure.

        The pipeline is:
        1. Keep base structure.
        2. Generate strained structures based on `distortion.strain_range` and `n_strain_steps`.
        3. For every structure produced so far (Base + Strained), generate `n_rattle_steps` rattled versions.

        Args:
            base_structure (Atoms): The starting structure (e.g., an SQS).

        Returns:
            List[Atoms]: A list containing the base, strained, and rattled structures.
                         If distortions are disabled, returns only the base structure.

        Raises:
            GeneratorError: If batch generation fails.
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
            structures_to_rattle = [base_structure, *strained_structures]
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
            msg = f"Batch generation failed: {e}"
            raise GeneratorError(msg) from e
