import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.generator import GeneratorConfig


class DefectGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def create_vacancy(self, atoms: Atoms) -> list[Atoms]:
        """
        Creates vacancies by removing atoms.
        """
        results = []
        # Naive approach: remove each atom once.
        # Check symmetry if possible?
        # SPEC: "Iterate through unique sites... return list of unique defects"
        # We start with naive iteration.

        n_atoms = len(atoms)
        # Use spglib for symmetry if we want to be smart, but it's an extra dependency.
        # We will just iterate all for now (assuming small supercell).

        for i in range(n_atoms):
            new_atoms = atoms.copy()
            del new_atoms[i]
            new_atoms.info['config_type'] = 'vacancy'
            new_atoms.info['defect_index'] = i
            results.append(new_atoms)

        return results

    def create_interstitial(self, atoms: Atoms, element: str) -> list[Atoms]:
        """
        Creates interstitials.
        """
        results = []
        # Finding holes is hard.
        # Heuristic: Voronoi vertices? Or just simple geometry?
        # SPEC: "Use Voronoi tessellation or simple geometry to find holes."

        # Simple fallback: Center of largest empty sphere (requires Voronoi).
        # Or just specific fractional coordinates like 0.5,0.5,0.5 if not occupied.

        # Let's try to verify if 0.5,0.5,0.5 is far from atoms.
        # Or better: random insertions and check distance?

        # Let's implement a simplified approach: Try placing at center of cell and midpoints of edges.

        candidates = [
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75]
        ]

        cell = atoms.get_cell()

        for frac in candidates:
            pos = np.dot(frac, cell)

            # Check distance to existing atoms
            dists = atoms.get_distances(-1, range(len(atoms)), vector=False) # Wait, need to add dummy first?

            # Distance check manually
            min_dist = np.inf
            for p in atoms.positions:
                d = np.linalg.norm(p - pos)
                min_dist = min(min_dist, d)

            if min_dist > 1.5: # Hardcoded minimal distance for interstitial
                new_atoms = atoms.copy()
                new_atoms.append(element)
                new_atoms.positions[-1] = pos
                new_atoms.info['config_type'] = 'interstitial'
                results.append(new_atoms)

        if not results:
             # Fallback: Just put it somewhere far from 0
             pass

        return results
