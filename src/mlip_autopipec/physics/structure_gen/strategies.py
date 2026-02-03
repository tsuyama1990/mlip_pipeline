import numpy as np
from ase import Atoms


class StrainGenerator:
    """Generates strained structures."""

    def __init__(self, strain_range: float = 0.1) -> None:
        self.strain_range = strain_range

    def generate(self, atoms: Atoms, count: int) -> list[Atoms]:
        candidates = []
        for _ in range(count):
            # Strain matrix epsilon
            eps = np.random.uniform(-self.strain_range, self.strain_range, (3, 3))
            # Symmetrize
            eps = (eps + eps.T) / 2.0
            deformation = np.eye(3) + eps

            new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
            # Deform cell
            # new_cell = old_cell @ deformation
            new_cell = atoms.cell @ deformation
            new_atoms.set_cell(new_cell, scale_atoms=True)
            candidates.append(new_atoms)

        # Sort by volume
        candidates.sort(key=lambda a: a.get_volume())
        return candidates


class DefectGenerator:
    """Generates structures with defects."""

    def __init__(
        self, defect_type: str = "vacancy", supercell_dim: tuple[int, int, int] = (3, 3, 3)
    ) -> None:
        self.defect_type = defect_type
        self.supercell_dim = supercell_dim

    def generate(self, atoms: Atoms, count: int) -> list[Atoms]:
        candidates = []
        for _ in range(count):
            # 1. Supercell
            # If small, make supercell
            if len(atoms) < 20:
                supercell = atoms * self.supercell_dim
            else:
                supercell = atoms.copy()  # type: ignore[no-untyped-call]

            # 2. Defect
            if self.defect_type == "vacancy":
                # Pick random atom
                idx = np.random.randint(0, len(supercell))
                del supercell[idx]

            candidates.append(supercell)

        return candidates
