"""Tests for physics validation logic."""

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import GPa

# We import from the module we are about to create.
# Since it doesn't exist yet, this will fail if run, which is expected for TDD.
# However, to write the test file without ImportErrors blocking pytest collection of *other* tests (if I were running them),
# I usually would need the module to exist.
# But here I am creating the test first. I will mock the imports or just expect failure.
# To satisfy the linter/mypy, I need to ignore the import error or ensure the module exists (at least empty).
# I will create empty physics.py in the next step (Logic Implementation), but for now I can just define the test.
# Mypy will complain about missing module. I will ignore it.
from pyacemaker.validator.physics import (
    check_elastic,
    check_eos,
    check_phonons,
)


@pytest.fixture
def mock_atoms() -> Atoms:
    """Create a mock Atoms object."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)
    atoms.calc = MagicMock(spec=Calculator)
    return atoms


def test_check_phonons_stable(mock_atoms: Atoms) -> None:
    """Test phonon stability check with stable frequencies."""
    with patch("pyacemaker.validator.physics.Phonopy") as MockPhonopy:
        instance = MockPhonopy.return_value
        # Mock band structure with positive frequencies
        # phonopy.get_band_structure_dict() returns a dict with 'frequencies' and 'qpoints' etc.
        instance.get_band_structure_dict.return_value = {
            "frequencies": [[0.0, 0.1, 1.0, 2.0], [0.0, 0.2, 1.5, 2.5]],
            "qpoints": [[0, 0, 0], [0.5, 0.5, 0.5]],
        }

        is_stable = check_phonons(mock_atoms, supercell=[2, 2, 2])
        assert is_stable is True


def test_check_phonons_unstable(mock_atoms: Atoms) -> None:
    """Test phonon stability check with imaginary frequencies."""
    with patch("pyacemaker.validator.physics.Phonopy") as MockPhonopy:
        instance = MockPhonopy.return_value
        # Mock band structure with imaginary (negative) frequencies
        instance.get_band_structure_dict.return_value = {
            "frequencies": [[0.0, -0.1, 1.0, 2.0], [0.0, 0.2, 1.5, 2.5]],
            "qpoints": [[0, 0, 0], [0.5, 0.5, 0.5]],
        }

        is_stable = check_phonons(mock_atoms, supercell=[2, 2, 2])
        assert is_stable is False


def test_check_eos_valid(mock_atoms: Atoms) -> None:
    """Test EOS check returns valid bulk modulus."""
    # Mock EquationOfState
    with patch("pyacemaker.validator.physics.EquationOfState") as MockEOS:
        instance = MockEOS.return_value
        # Mock fit return value (v0, e0, B, dB/dP)
        # B is returned in eV/A^3 by fit(), check_eos converts to GPa by dividing by GPa
        # So if we want result 150.0 GPa, we should return 150.0 * GPa
        instance.fit.return_value = (100.0, -10.0, 150.0 * GPa, 4.0)

        # Mock plot to prevent actual plotting
        instance.plot.return_value = MagicMock()

        bulk_modulus, plot_path = check_eos(mock_atoms, strain=0.05)

        assert bulk_modulus == pytest.approx(150.0)
        assert plot_path == "eos.png"


def test_check_elastic_stable(mock_atoms: Atoms) -> None:
    """Test elastic stability check."""
    # Mock elastic constants calculation
    with patch("pyacemaker.validator.physics.calculate_elastic_constants") as mock_calc:
        # Mock Cij matrix for cubic system (stable)
        # C11 > |C12|, C11 + 2C12 > 0, C44 > 0
        mock_calc.return_value = {"C11": 160.0, "C12": 60.0, "C44": 80.0}

        is_stable, Cij = check_elastic(mock_atoms)

        assert is_stable is True
        assert Cij["C11"] == 160.0


def test_check_elastic_unstable(mock_atoms: Atoms) -> None:
    """Test elastic stability check for unstable system."""
    with patch("pyacemaker.validator.physics.calculate_elastic_constants") as mock_calc:
        # Unstable case: C11 < C12
        mock_calc.return_value = {"C11": 50.0, "C12": 60.0, "C44": 80.0}

        is_stable, Cij = check_elastic(mock_atoms)

        assert is_stable is False
