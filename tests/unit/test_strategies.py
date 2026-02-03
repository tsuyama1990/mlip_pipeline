import numpy as np
from ase.build import bulk

from mlip_autopipec.physics.structure_gen.strategies import (
    DefectGenerator,
    RandomDisplacementGenerator,
    StrainGenerator,
)


def test_strain_generator_basics() -> None:
    atoms = bulk("Cu", "fcc", a=3.6)
    # Range 0.1 (10%)
    gen = StrainGenerator(strain_range=0.1)

    candidates = gen.generate(atoms, count=5)
    assert len(candidates) == 5

    # Check if volumes vary
    vols = [a.get_volume() for a in candidates]  # type: ignore[no-untyped-call]
    # At least some variation should exist
    assert np.std(vols) > 0.0


def test_defect_generator_vacancy() -> None:
    # Make a supercell so vacancy is not removing the whole crystal
    # Use (3,3,3) -> 54 atoms (>20) to avoid auto-supercell in generator
    atoms = bulk("Si", "diamond", a=5.43) * (3, 3, 3)
    original_count = len(atoms)
    assert original_count > 20

    gen = DefectGenerator(defect_type="vacancy")
    candidates = gen.generate(atoms, count=1)

    assert len(candidates) == 1
    defect_struct = candidates[0]

    # Expect 1 atom less
    assert len(defect_struct) == original_count - 1


def test_random_displacement_generator() -> None:
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    original_positions = atoms.positions.copy()

    displacement = 0.05
    gen = RandomDisplacementGenerator(displacement_range=displacement)
    count = 10
    candidates = gen.generate(atoms, count=count)

    assert len(candidates) == count

    for cand in candidates:
        # Check positions are different
        assert not np.allclose(cand.positions, original_positions)
        # Check displacement magnitude is within range
        diff = cand.positions - original_positions
        assert np.all(np.abs(diff) <= displacement)
