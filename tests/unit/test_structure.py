import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure


def test_structure_valid_creation() -> None:
    pos = np.zeros((2, 3))
    numbers = np.array([1, 1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])

    s = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
    assert np.allclose(s.positions, pos)
    assert np.allclose(s.atomic_numbers, numbers)


def test_structure_invalid_positions() -> None:
    with pytest.raises(ValueError, match="Positions must be"):
        Structure(
            positions=np.zeros((2, 2)),  # Wrong shape
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )


def test_structure_mismatch() -> None:
    with pytest.raises(ValueError, match="Mismatch"):
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1]),  # Mismatch
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )


def test_serialization() -> None:
    pos = np.zeros((1, 3))
    numbers = np.array([6])
    cell = np.eye(3)
    pbc = np.array([True, True, True])

    s = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
    json_str = s.model_dump_json()
    s2 = Structure.model_validate_json(json_str)

    assert np.allclose(s.positions, s2.positions)


def test_from_ase() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
    s = Structure.from_ase(atoms)
    assert len(s.positions) == 2
    assert s.atomic_numbers[0] == 1
    assert np.allclose(s.cell, np.array(atoms.get_cell()))  # type: ignore[no-untyped-call]


def test_forces_validation() -> None:
    pos = np.zeros((2, 3))
    numbers = np.array([1, 1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])
    forces = np.zeros((2, 3))

    s = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc, forces=forces)
    assert s.forces is not None
    assert np.allclose(s.forces, forces)

    with pytest.raises(ValueError, match="Forces must be"):
        Structure(
            positions=pos,
            atomic_numbers=numbers,
            cell=cell,
            pbc=pbc,
            forces=np.zeros((2, 2)),  # Wrong shape
        )


def test_structure_stress_validation() -> None:
    pos = np.zeros((2, 3))
    numbers = np.array([1, 1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])
    stress = np.zeros(6)  # Voigt

    s = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc, stress=stress)
    assert s.stress is not None
    assert s.stress.shape == (6,)

    stress_tensor = np.zeros((3, 3))
    s2 = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc, stress=stress_tensor)
    assert s2.stress is not None
    assert s2.stress.shape == (3, 3)

    with pytest.raises(ValueError, match="Stress must be"):
        Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc, stress=np.zeros(5))


def test_to_ase() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    numbers = np.array([1, 1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])
    energy = -10.0
    forces = np.zeros((2, 3))
    stress = np.zeros(6)

    s = Structure(
        positions=pos,
        atomic_numbers=numbers,
        cell=cell,
        pbc=pbc,
        energy=energy,
        forces=forces,
        stress=stress,
    )

    atoms = s.to_ase()
    assert len(atoms) == 2
    assert atoms.info["energy"] == energy
    assert np.allclose(atoms.arrays["forces"], forces)
    assert np.allclose(atoms.info["stress"], stress)


def test_physical_validation() -> None:
    pos = np.zeros((1, 3))
    cell = np.eye(3)
    pbc = np.array([True, True, True])

    # Invalid atomic number
    with pytest.raises(ValueError, match="Atomic numbers must be between 1 and 118"):
        Structure(positions=pos, atomic_numbers=np.array([119]), cell=cell, pbc=pbc)

    # Infinite energy
    with pytest.raises(ValueError, match="Energy must be finite"):
        Structure(
            positions=pos, atomic_numbers=np.array([1]), cell=cell, pbc=pbc, energy=float("inf")
        )

    # Large energy
    with pytest.raises(ValueError, match="Energy magnitude exceeds reasonable limit"):
        Structure(positions=pos, atomic_numbers=np.array([1]), cell=cell, pbc=pbc, energy=2e6)

    # Large forces
    with pytest.raises(ValueError, match="Forces magnitude exceeds reasonable limit"):
        Structure(
            positions=pos,
            atomic_numbers=np.array([1]),
            cell=cell,
            pbc=pbc,
            forces=np.array([[2000.0, 0.0, 0.0]]),
        )


def test_validate_labeled() -> None:
    pos = np.zeros((1, 3))
    numbers = np.array([1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])

    # Missing labels
    s = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
    with pytest.raises(ValueError, match="Structure missing energy label"):
        s.validate_labeled()

    s.energy = -1.0
    with pytest.raises(ValueError, match="Structure missing forces label"):
        s.validate_labeled()

    s.forces = np.zeros((1, 3))
    with pytest.raises(ValueError, match="Structure missing stress label"):
        s.validate_labeled()

    s.stress = np.zeros(6)
    s.validate_labeled()  # Should pass


def test_validate_labeled_partial() -> None:
    """Test validate_labeled with various combinations of missing labels."""
    pos = np.zeros((1, 3))
    numbers = np.array([1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])

    # Only Energy
    s = Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc, energy=-1.0)
    with pytest.raises(ValueError, match="Structure missing forces label"):
        s.validate_labeled()

    # Energy and Stress (missing Forces)
    s.stress = np.zeros(6)
    with pytest.raises(ValueError, match="Structure missing forces label"):
        s.validate_labeled()

    # Forces and Stress (missing Energy)
    s = Structure(
        positions=pos,
        atomic_numbers=numbers,
        cell=cell,
        pbc=pbc,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6)
    )
    with pytest.raises(ValueError, match="Structure missing energy label"):
        s.validate_labeled()
