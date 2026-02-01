from typing import List, Literal
import numpy as np
import ase
from mlip_autopipec.domain_models.structure import Structure

class DefectStrategy:
    """
    Strategy for generating defect structures (vacancies, interstitials, antisites).
    See SPEC.md Section 3.2.
    """

    def apply(
        self,
        structure: Structure,
        defect_type: Literal["vacancy", "interstitial", "antisite"] = "vacancy",
        count: int = 1
    ) -> List[Structure]:
        """
        Apply defect generation strategy to the input structure.

        Args:
            structure: Base structure.
            defect_type: Type of defect to introduce.
            count: Number of variations to generate (e.g. remove different atoms).

        Returns:
            List of modified Structures.
        """
        atoms = structure.to_ase()
        results = []

        if defect_type == "vacancy":
            # Generate 'count' structures, each with 1 vacancy at different random positions
            indices = list(range(len(atoms)))
            # If structure is small, we might duplicate, so limit count
            n_samples = min(count, len(atoms))

            # Select random indices to remove
            # For reproducibility, we might want a seed, but usually these are stochastic
            # Let's pick random indices
            targets = np.random.choice(indices, size=n_samples, replace=False)

            for idx in targets:
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                del new_atoms[int(idx)]
                results.append(Structure.from_ase(new_atoms))

        elif defect_type == "interstitial":
            # Add an atom. Ideally at a void.
            # Simple approach: Identify "largest empty sphere" or just random placement.
            # Or assume 0,0,0 relative if not occupied?
            # Let's try to place it in a reasonably open spot.
            # For simplicity in this iteration: Place at center of longest bond or random fractional coordinate.

            for _ in range(count):
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                # Random fractional coordinate
                pos = np.random.rand(3)
                # Determine species: pick one from existing
                species = new_atoms.get_chemical_symbols() # type: ignore[no-untyped-call]
                element = species[np.random.randint(len(species))]

                # Check distance to neighbors to avoid too close overlap (optional but good)
                # We skip complex check for now and rely on relaxation/minimization later if needed.
                # But to avoid exploding simulation, we should be careful.
                # UAT says "We want to ensure potential learns about vacancies". Interstitials are bonus.

                new_atoms.append(ase.Atom(element, position=new_atoms.get_cell().dot(pos))) # type: ignore[no-untyped-call]
                results.append(Structure.from_ase(new_atoms))

        elif defect_type == "antisite":
            # Swap two atoms of different species
            species = atoms.get_chemical_symbols() # type: ignore[no-untyped-call]
            unique_species = list(set(species))

            if len(unique_species) < 2:
                # Cannot create antisite in elemental crystal
                # Return empty list or just return original?
                # Let's return empty to indicate failure to apply strategy
                return []

            indices = list(range(len(atoms)))

            for _ in range(count):
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                # Pick two indices with different species
                # Attempt 10 times to find a pair
                for _retry in range(10):
                    idx1, idx2 = np.random.choice(indices, 2, replace=False)
                    if new_atoms[idx1].symbol != new_atoms[idx2].symbol:
                        # Swap
                        s1 = new_atoms[idx1].symbol
                        s2 = new_atoms[idx2].symbol
                        new_atoms[idx1].symbol = s2
                        new_atoms[idx2].symbol = s1
                        results.append(Structure.from_ase(new_atoms))
                        break

        return results
