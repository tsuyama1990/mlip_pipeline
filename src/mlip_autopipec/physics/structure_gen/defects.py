from typing import Iterator, Literal, Optional
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
        count: int = 1,
        seed: Optional[int] = None
    ) -> Iterator[Structure]:
        """
        Apply defect generation strategy to the input structure.

        Args:
            structure: Base structure.
            defect_type: Type of defect to introduce.
            count: Number of variations to generate (e.g. remove different atoms).
            seed: Random seed for reproducibility.

        Returns:
            Iterator yielding modified Structures.
        """
        rng = np.random.default_rng(seed)
        atoms = structure.to_ase()

        if defect_type == "vacancy":
            # Generate 'count' structures, each with 1 vacancy at different random positions
            indices = list(range(len(atoms)))
            n_samples = min(count, len(atoms))

            # Use rng.choice
            targets = rng.choice(indices, size=n_samples, replace=False)

            for idx in targets:
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                del new_atoms[int(idx)]
                yield Structure.from_ase(new_atoms)

        elif defect_type == "interstitial":
            for _ in range(count):
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                # Random fractional coordinate
                pos = rng.random(3)
                species = new_atoms.get_chemical_symbols() # type: ignore[no-untyped-call]
                element = species[rng.integers(len(species))]

                new_atoms.append(ase.Atom(element, position=new_atoms.get_cell().dot(pos))) # type: ignore[no-untyped-call]
                yield Structure.from_ase(new_atoms)

        elif defect_type == "antisite":
            # Swap two atoms of different species
            species = atoms.get_chemical_symbols() # type: ignore[no-untyped-call]
            unique_species = list(set(species))

            if len(unique_species) < 2:
                return

            indices = list(range(len(atoms)))

            for _ in range(count):
                new_atoms = atoms.copy() # type: ignore[no-untyped-call]
                # Pick two indices with different species
                # Attempt 10 times to find a pair
                for _retry in range(10):
                    idx1, idx2 = rng.choice(indices, 2, replace=False)
                    if new_atoms[idx1].symbol != new_atoms[idx2].symbol:
                        # Swap
                        s1 = new_atoms[idx1].symbol
                        s2 = new_atoms[idx2].symbol
                        new_atoms[idx1].symbol = s2
                        new_atoms[idx2].symbol = s1
                        yield Structure.from_ase(new_atoms)
                        break
