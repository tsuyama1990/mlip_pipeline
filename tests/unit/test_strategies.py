import numpy as np
from ase import Atoms

from mlip_autopipec.physics.structure_gen.strategies import (
    DefectGenerator,
    RandomDisplacementGenerator,
    StrainGenerator,
)


def test_random_displacement_generator() -> None:
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True)
    generator = RandomDisplacementGenerator(displacement_range=0.1)

    candidates = generator.generate(atoms, count=5)

    assert len(candidates) == 5
    for cand in candidates:
        assert len(cand) == 2
        # Check positions are different but close
        dist = np.linalg.norm(cand.get_positions() - atoms.get_positions())  # type: ignore[no-untyped-call]
        assert dist > 0
        assert dist < 0.2 * np.sqrt(2) * 2  # Rough check, max displacement per atom


def test_strain_generator() -> None:
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True)
    generator = StrainGenerator(strain_range=0.1)

    candidates = generator.generate(atoms, count=3)
    assert len(candidates) == 3
    for cand in candidates:
        # Check cell is changed
        assert not np.allclose(cand.get_cell(), atoms.get_cell())  # type: ignore[no-untyped-call]


def test_defect_generator() -> None:
    # 2x2x2 supercell of 2 atoms = 16 atoms
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
    generator = DefectGenerator(defect_type="vacancy", supercell_dim=(2, 2, 2))

    candidates = generator.generate(atoms, count=2)
    assert len(candidates) == 2
    for cand in candidates:
        # Should have 16 - 1 = 15 atoms
        assert len(cand) == 15
