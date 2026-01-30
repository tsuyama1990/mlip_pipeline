import ase.build
import numpy as np
from mlip_autopipec.domain_models.structure import Structure


class ColdStartStrategy:
    def generate(
        self,
        element: str,
        crystal_structure: str,
        lattice_constant: float,
        supercell: tuple[int, int, int],
    ) -> Structure:
        # Use ase.build.bulk
        # We enforce cubic=True to get conventional cell
        atoms = ase.build.bulk(
            element, crystalstructure=crystal_structure, a=lattice_constant, cubic=True
        )  # type: ignore[no-untyped-call]

        # Supercell
        if supercell != (1, 1, 1):
            atoms = atoms * supercell

        return Structure.from_ase(atoms)


class RattleStrategy:
    def __init__(self, stdev: float, seed: int):
        self.stdev = stdev
        self.rng = np.random.default_rng(seed)

    def apply(self, structure: Structure) -> Structure:
        if self.stdev <= 0:
            return structure

        atoms = structure.to_ase()
        # ase.Atoms.rattle expects an int seed or None.
        # We generate a deterministic int from our RNG.
        seed = int(self.rng.integers(0, 2**31 - 1))
        atoms.rattle(stdev=self.stdev, seed=seed)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)
