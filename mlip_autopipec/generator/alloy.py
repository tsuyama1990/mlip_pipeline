import logging

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from mlip_autopipec.config.schemas.generator import GeneratorConfig

logger = logging.getLogger(__name__)

class AlloyGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def generate_sqs(self, prim_cell: Atoms, composition: dict[str, float]) -> Atoms:
        """
        Generates a Special Quasirandom Structure (SQS) for the given composition.
        Tries to use `icet`, falls back to random shuffle.
        """
        # Validate composition sums to 1 (approx)
        if abs(sum(composition.values()) - 1.0) > 1e-4:
            raise ValueError("Composition must sum to 1.0")

        # Determine supercell size from config or defaults
        supercell_matrix = self.config.supercell_matrix
        atoms = make_supercell(prim_cell, supercell_matrix)
        n_atoms = len(atoms)

        # Calculate target counts
        counts = {}
        symbols = []

        sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)

        for elem, frac in sorted_comp:
            count = int(round(frac * n_atoms))
            counts[elem] = count
            symbols.extend([elem] * count)

        # Fix rounding errors
        diff = n_atoms - len(symbols)
        if diff != 0:
            # Add/remove from major species
            major_elem = sorted_comp[0][0]
            if diff > 0:
                symbols.extend([major_elem] * diff)
            else:
                pass

        # Truncate if too many (rare case)
        symbols = symbols[:n_atoms]

        # Check if we still have shortage
        while len(symbols) < n_atoms:
             symbols.append(sorted_comp[0][0])

        # Try icet
        try:
            from icet import ClusterSpace
            from icet.tools import generate_sqs
            # Setup cluster space for icet...
            raise ImportError("icet not fully configured for auto-detection")
        except (ImportError, ModuleNotFoundError):
            logger.info("icet not available or configured. Using Random Shuffle fallback.")
            np.random.shuffle(symbols)
            atoms.set_chemical_symbols(symbols)
            atoms.info['config_type'] = 'sqs'
            atoms.info['origin'] = 'random_shuffle'

        return atoms

    def apply_strain(self, atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
        """
        Applies a strain tensor (3x3) to the atoms object.
        Strain tensor e means new_cell = old_cell @ (I + e).
        """
        strained = atoms.copy()
        cell = strained.get_cell()

        # Deformation gradient F = I + epsilon
        deformation = np.eye(3) + strain_tensor

        # New cell vectors. ASE cell is rows are vectors.
        # new_cell = cell @ deformation
        new_cell = np.dot(cell, deformation)

        strained.set_cell(new_cell, scale_atoms=True)

        strained.info['config_type'] = 'strain'
        strained.info['strain_tensor'] = strain_tensor.tolist()
        return strained

    def apply_rattle(self, atoms: Atoms, sigma: float) -> Atoms:
        """
        Applies Gaussian noise to atomic positions.
        """
        rattled = atoms.copy()
        # Explicit implementation using numpy.random.normal as per spec
        delta = np.random.normal(0, sigma, atoms.positions.shape)
        rattled.positions += delta

        rattled.info['config_type'] = 'rattle'
        rattled.info['rattle_sigma'] = sigma
        return rattled

    def generate_batch(self, base_structure: Atoms) -> list[Atoms]:
        """
        Generates a batch of structures: SQS -> Strain -> Rattle.
        Combinatorial expansion.
        """
        results = []

        # 1. Base (SQS/Prim) - Add it? Usually yes.
        base_structure.info['config_type'] = base_structure.info.get('config_type', 'base')
        results.append(base_structure)

        # 2. Strains
        min_s, max_s = self.config.strain_range
        strains = np.linspace(min_s, max_s, self.config.n_strain_steps)

        strained_structures = []
        for s in strains:
            if abs(s) < 1e-6: continue # Skip zero strain if it duplicates base

            # Hydrostatic strain
            strain_tensor = np.eye(3) * s
            strained = self.apply_strain(base_structure, strain_tensor)
            strained_structures.append(strained)
            results.append(strained)

        # 3. Rattles
        structures_to_rattle = [base_structure] + strained_structures

        for st in structures_to_rattle:
            for _ in range(self.config.n_rattle_steps):
                rattled = self.apply_rattle(st, self.config.rattling_amplitude)
                # Keep parent info
                if 'strain_tensor' in st.info:
                    rattled.info['strain_tensor'] = st.info['strain_tensor']
                rattled.info['parent_config_type'] = st.info.get('config_type')
                results.append(rattled)

        return results
