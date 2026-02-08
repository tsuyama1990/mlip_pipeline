import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure

DUMMY_POSITIONS = np.zeros((2, 3))
DUMMY_ATOMIC_NUMBERS = np.array([1, 1])
DUMMY_CELL = np.eye(3)
DUMMY_PBC = np.array([True, True, True])
VOIGT_STRESS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
EXPECTED_TENSOR_STRESS = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])


def test_structure_valid_creation() -> None:
    s = Structure(
        positions=DUMMY_POSITIONS,
        atomic_numbers=DUMMY_ATOMIC_NUMBERS,
        cell=DUMMY_CELL,
        pbc=DUMMY_PBC,
    )
    assert np.allclose(s.positions, DUMMY_POSITIONS)
    assert np.allclose(s.atomic_numbers, DUMMY_ATOMIC_NUMBERS)


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
    assert np.allclose(s.cell, np.array(atoms.get_cell()))


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
    # Voigt input should be converted to 3x3
    s = Structure(
        positions=DUMMY_POSITIONS,
        atomic_numbers=DUMMY_ATOMIC_NUMBERS,
        cell=DUMMY_CELL,
        pbc=DUMMY_PBC,
        stress=VOIGT_STRESS,
    )
    assert s.stress is not None
    assert s.stress.shape == (3, 3)
    assert np.allclose(s.stress, EXPECTED_TENSOR_STRESS)

    stress_tensor = np.zeros((3, 3))
    s2 = Structure(
        positions=DUMMY_POSITIONS,
        atomic_numbers=DUMMY_ATOMIC_NUMBERS,
        cell=DUMMY_CELL,
        pbc=DUMMY_PBC,
        stress=stress_tensor,
    )
    assert s2.stress is not None
    assert s2.stress.shape == (3, 3)

    with pytest.raises(ValueError, match="Stress must be"):
        Structure(
            positions=DUMMY_POSITIONS,
            atomic_numbers=DUMMY_ATOMIC_NUMBERS,
            cell=DUMMY_CELL,
            pbc=DUMMY_PBC,
            stress=np.zeros(5),
        )


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

    # Check labels via calculator
    assert atoms.calc is not None
    assert atoms.get_potential_energy() == energy
    assert np.allclose(atoms.get_forces(), forces)

    # Stress should be converted to 3x3 in Structure and preserved in Calculator
    expected_stress = np.zeros((3, 3))
    expected_stress[0, 0] = stress[0]
    expected_stress[1, 1] = stress[1]
    expected_stress[2, 2] = stress[2]
    # Structure.voigt_6_to_full_3x3 handles conversion logic (tested elsewhere)
    # Here we just verify it matches what we expect from input

    # SinglePointCalculator.get_stress(voigt=False) returns 3x3
    assert np.allclose(atoms.get_stress(voigt=False), expected_stress)
    assert atoms.get_stress(voigt=False).shape == (3, 3)


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
        stress=np.zeros(6),
    )
    with pytest.raises(ValueError, match="Structure missing energy label"):
        s.validate_labeled()

def test_structure_copy() -> None:
    # Create structure with all fields
    pos = np.array([[0.0, 0.0, 0.0]])
    numbers = np.array([1])
    cell = np.eye(3)
    pbc = np.array([True, True, True])
    forces = np.array([[0.1, 0.2, 0.3]])
    energy = -1.5
    stress = np.eye(3) * 0.1
    uncertainty = 0.5
    tags = {"type": "bulk", "provenance": {"id": 1}}

    s = Structure(
        positions=pos,
        atomic_numbers=numbers,
        cell=cell,
        pbc=pbc,
        forces=forces,
        energy=energy,
        stress=stress,
        uncertainty=uncertainty,
        tags=tags,
    )

    s_copy = s.model_deep_copy()

    # Check equality
    assert np.allclose(s_copy.positions, s.positions)
    assert np.allclose(s_copy.forces, s.forces)
    assert s_copy.tags == s.tags
    assert s_copy.energy == s.energy
    assert s_copy.uncertainty == s.uncertainty

    # Check deep copy (independence)
    s_copy.positions[0, 0] = 99.0
    assert s.positions[0, 0] == 0.0  # Original should not change

    s_copy.tags["type"] = "surface"
    assert s.tags["type"] == "bulk" # Original should not change

    s_copy.tags["provenance"]["id"] = 2
    assert s.tags["provenance"]["id"] == 1 # Deep copy of nested dict
