from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.validator.phonon import PhononAnalyzer, PhononResults


def test_phonon_analyzer_success() -> None:
    """Test PhononAnalyzer with successful phonopy run."""
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5.43, 5.43, 5.43], pbc=True)
    potential = MagicMock(spec=Potential)
    potential.path = Path("mock.pot")

    # Mock Phonopy
    with patch("mlip_autopipec.validator.phonon.Phonopy") as MockPhonopy, \
         patch("mlip_autopipec.validator.phonon.PhonopyAtoms"), \
         patch("mlip_autopipec.validator.phonon.MLIPCalculatorFactory") as MockFactory:

        mock_phonopy_instance = MockPhonopy.return_value

        # Mock supercells
        mock_sc = MagicMock()
        mock_sc.symbols = ["Si"]
        mock_sc.cell = np.eye(3) * 5.43
        mock_sc.scaled_positions = [[0,0,0]]

        mock_phonopy_instance.supercells_with_displacements = [mock_sc]

        # Mock frequencies: all real (positive)
        # get_mesh returns (qpoints, weights, frequencies, eigenvectors)
        # frequencies shape (nq, nband)
        mock_phonopy_instance.run_mesh.return_value = None
        mock_phonopy_instance.get_mesh_dict.return_value = {
            "frequencies": [[0.1, 1.0, 2.0], [0.2, 1.1, 2.1]]
        }

        # Also need to mock create() returning a calc that works with get_forces()
        mock_calc = MagicMock()
        MockFactory.return_value.create.return_value = mock_calc

        analyzer = PhononAnalyzer(supercell_matrix=[2, 2, 2])
        results = analyzer.calculate_phonons(atoms, potential)

        assert isinstance(results, PhononResults)
        assert results.is_stable is True
        assert results.max_imaginary_freq == 0.0

def test_phonon_analyzer_unstable() -> None:
    """Test PhononAnalyzer with imaginary frequencies."""
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5.43, 5.43, 5.43], pbc=True)
    potential = MagicMock(spec=Potential)
    potential.path = Path("mock.pot")

    with patch("mlip_autopipec.validator.phonon.Phonopy") as MockPhonopy, \
         patch("mlip_autopipec.validator.phonon.PhonopyAtoms"), \
         patch("mlip_autopipec.validator.phonon.MLIPCalculatorFactory") as MockFactory:

        mock_phonopy_instance = MockPhonopy.return_value

        mock_sc = MagicMock()
        mock_sc.symbols = ["Si"]
        mock_sc.cell = np.eye(3) * 5.43
        mock_sc.scaled_positions = [[0,0,0]]
        mock_phonopy_instance.supercells_with_displacements = [mock_sc]

        mock_calc = MagicMock()
        MockFactory.return_value.create.return_value = mock_calc

        # Imaginary frequencies (negative)
        mock_phonopy_instance.run_mesh.return_value = None
        mock_phonopy_instance.get_mesh_dict.return_value = {
            "frequencies": [[-0.5, 1.0, 2.0], [0.2, 1.1, 2.1]]
        }

        analyzer = PhononAnalyzer()
        results = analyzer.calculate_phonons(atoms, potential)

        assert results.is_stable is False
        assert results.max_imaginary_freq == pytest.approx(0.5)

def test_phonon_analyzer_no_phonopy() -> None:
    """Test behavior when phonopy is missing (ImportError)."""
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5.43, 5.43, 5.43], pbc=True)
    potential = MagicMock(spec=Potential)
    potential.path = Path("mock.pot")

    # Patch _PHONOPY_AVAILABLE to False
    with patch("mlip_autopipec.validator.phonon._PHONOPY_AVAILABLE", False):
        analyzer = PhononAnalyzer()
        results = analyzer.calculate_phonons(atoms, potential)

        assert results.is_stable is True
