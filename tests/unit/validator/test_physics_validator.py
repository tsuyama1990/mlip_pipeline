"""Tests for PhysicsValidator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.validator.physics import PhysicsValidator


@pytest.fixture
def physics_validator() -> PhysicsValidator:
    """Fixture for PhysicsValidator."""
    return PhysicsValidator()


@pytest.fixture
def mock_atoms() -> Atoms:
    """Fixture for mock atoms."""
    atoms = MagicMock(spec=Atoms)
    atoms.calc = MagicMock()
    atoms.get_chemical_symbols.return_value = ["Si"] * 2
    atoms.get_scaled_positions.return_value = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    # Use numpy array for cell
    atoms.cell = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    atoms.get_cell.return_value = atoms.cell.copy()
    atoms.copy.return_value = atoms
    atoms.__len__.return_value = 2  # Not empty
    return atoms


def test_check_phonons_success(physics_validator: PhysicsValidator, mock_atoms: Atoms) -> None:
    """Test phonon check success."""
    with patch("pyacemaker.validator.physics.Phonopy") as mock_phonopy_cls:
        mock_phonopy = mock_phonopy_cls.return_value
        mock_phonopy.produce_force_constants.return_value = None
        mock_phonopy.get_band_structure_dict.return_value = {
            "frequencies": [np.array([1.0, 2.0]), np.array([1.0, 2.0])]
        }
        mock_phonopy.get_supercells_with_displacements.return_value = [
            MagicMock(symbols=["Si"], cell=np.eye(3), scaled_positions=np.zeros((1, 3)))
        ]

        result = physics_validator.check_phonons(mock_atoms)

        assert result is True
        # Verify force constants were produced with external forces
        mock_phonopy.produce_force_constants.assert_called_once()
        call_kwargs = mock_phonopy.produce_force_constants.call_args.kwargs
        assert "forces" in call_kwargs or len(mock_phonopy.produce_force_constants.call_args.args) > 0


def test_check_phonons_failure(physics_validator: PhysicsValidator, mock_atoms: Atoms) -> None:
    """Test phonon check failure (imaginary frequencies)."""
    with patch("pyacemaker.validator.physics.Phonopy") as mock_phonopy_cls:
        mock_phonopy = mock_phonopy_cls.return_value
        mock_phonopy.get_band_structure_dict.return_value = {
            "frequencies": [np.array([-1.0, 2.0]), np.array([1.0, 2.0])]
        }
        mock_phonopy.get_supercells_with_displacements.return_value = [
             MagicMock(symbols=["Si"], cell=np.eye(3), scaled_positions=np.zeros((1, 3)))
        ]

        result = physics_validator.check_phonons(mock_atoms, tolerance=0.0)
        assert result is False

def test_check_eos(physics_validator: PhysicsValidator, mock_atoms: Atoms) -> None:
    """Test EOS check."""
    with patch("pyacemaker.validator.physics.EquationOfState") as mock_eos_cls, \
         patch("pyacemaker.validator.physics.plt"):

        mock_eos = mock_eos_cls.return_value
        mock_eos.fit.return_value = (100.0, -10.0, 1.0, 4.0)

        B_GPa, plot_path = physics_validator.check_eos(mock_atoms)

        assert B_GPa > 0
        assert plot_path == "eos.png"

def test_check_elastic(physics_validator: PhysicsValidator, mock_atoms: Atoms) -> None:
    """Test elastic check."""
    with patch.object(physics_validator, "calculate_elastic_constants") as mock_calc:
        mock_calc.return_value = {"C11": 200.0, "C12": 100.0, "C44": 100.0}
        stable, Cij = physics_validator.check_elastic(mock_atoms)
        assert stable is True
        assert Cij["C11"] == 200.0

def test_check_eos_empty_structure(physics_validator: PhysicsValidator) -> None:
    """Test EOS check with empty structure."""
    atoms = MagicMock(spec=Atoms)
    atoms.calc = MagicMock()
    atoms.__len__.return_value = 0

    with pytest.raises(ValueError, match="empty structure"):
        physics_validator.check_eos(atoms)

def test_check_phonons_no_calculator(physics_validator: PhysicsValidator) -> None:
    """Test phonon check without calculator."""
    atoms = MagicMock(spec=Atoms)
    atoms.calc = None

    with pytest.raises(ValueError, match="calculator attached"):
        physics_validator.check_phonons(atoms)
