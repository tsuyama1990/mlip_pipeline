from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.validation import MetricResult

# These imports will fail until implementation is done
from mlip_autopipec.validation.metrics import ElasticValidator, PhononValidator


class TestElasticValidator:
    def test_calculate_stiffness(self) -> None:
        """Test internal stiffness calculation logic."""
        potential = MagicMock()
        structure = Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)

        # Mock ASE components
        with patch("mlip_autopipec.validation.metrics.LBFGS"), \
             patch("mlip_autopipec.validation.metrics.UnitCellFilter"):

            # Mock atoms.get_stress return values
            # The method calls get_stress 3 times:
            # 1. Relaxed (stress_0)
            # 2. Strain xx (stress_xx)
            # 3. Strain xy (stress_xy)

            # Stress in ASE is usually 6-element array [xx, yy, zz, yz, xz, xy] in eV/A^3
            # We want C11 ~ 170 GPa. 1 GPa approx 0.00624 eV/A^3.
            # 170 GPa ~ 1.06 eV/A^3.
            # eps = 0.01. delta_stress = C * eps ~ 0.0106.

            stress_0 = np.array([0.0]*6)
            stress_xx = np.array([0.0106, 0.0, 0.0, 0.0, 0.0, 0.0]) # delta in xx
            stress_xy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.005]) # delta in xy (index 5)

            # We need to mock structure.copy() because the code runs on copies
            # And structure.get_stress()
            # But wait, structure.copy() creates a NEW atoms object.
            # My patch on "ase.Atoms.get_stress" should work if it patches the class method.

            with patch("ase.Atoms.get_stress", side_effect=[stress_0, stress_xx, stress_xy]):
                 cij = ElasticValidator._calculate_stiffness(potential, structure)

                 assert "C11" in cij
                 assert "C12" in cij
                 assert "C44" in cij
                 assert cij["C11"] > 0

    def test_run_success(self) -> None:
        """Test that elastic validator returns passed result when criteria met."""
        potential = MagicMock()
        structure = Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)

        with patch("mlip_autopipec.validation.metrics.ElasticValidator._calculate_stiffness") as mock_calc:
            # Return stable cubic coefficients (approx Cu)
            # C11=168, C12=121, C44=75 (GPa)
            mock_calc.return_value = {
                "C11": 170.0,
                "C12": 120.0,
                "C44": 75.0,
            }

            result = ElasticValidator.run(potential, structure)

            assert isinstance(result, MetricResult)
            assert result.name == "Elastic Stability"
            assert result.passed is True
            assert "C11" in result.details["Cij"]

    def test_run_failure(self) -> None:
        """Test failure when stability criteria violated (e.g. C11 < 0)."""
        potential = MagicMock()
        structure = Atoms("Cu")

        with patch("mlip_autopipec.validation.metrics.ElasticValidator._calculate_stiffness") as mock_calc:
            mock_calc.return_value = {
                "C11": -100.0,
                "C12": 120.0,
                "C44": 75.0,
            }

            result = ElasticValidator.run(potential, structure)
            assert result.passed is False


class TestPhononValidator:
    def test_calculate_band_structure(self) -> None:
        """Test phonon calculation logic with mocks."""
        potential = MagicMock()
        structure = Atoms("Si", positions=[[0, 0, 0]], cell=[5.4, 5.4, 5.4], pbc=True)

        with patch("mlip_autopipec.validation.metrics.PHONOPY_AVAILABLE", True), \
             patch("mlip_autopipec.validation.metrics.phonopy") as mock_phonopy, \
             patch("mlip_autopipec.validation.metrics.PhonopyAtoms"):

            mock_inst = mock_phonopy.Phonopy.return_value
            mock_inst.get_supercells_with_displacements.return_value = []
            mock_inst.get_band_structure_dict.return_value = {'frequencies': [[1.0, 2.0], [1.5, 2.5]]}

            mock_phonopy.get_band_qpoints_and_path_connections.return_value = ([], [])

            min_freq = PhononValidator._calculate_band_structure(potential, structure)
            assert min_freq == 1.0

    def test_run_success(self) -> None:
        """Test phonon validator passes with no imaginary modes."""
        potential = MagicMock()
        structure = Atoms("Si")

        # We need to ensure PHONOPY_AVAILABLE is True for this test context if we want to run logic,
        # but here we mock _calculate_band_structure, so it doesn't matter unless we remove the mock.
        with patch("mlip_autopipec.validation.metrics.PhononValidator._calculate_band_structure") as mock_calc, \
             patch("mlip_autopipec.validation.metrics.PHONOPY_AVAILABLE", True):
            # Mock return of min frequency
            mock_calc.return_value = 0.5  # Positive frequency

            result = PhononValidator.run(potential, structure)
            assert result.passed is True
            assert result.score == 0.5

    def test_run_failure(self) -> None:
        """Test failure with imaginary modes (negative frequency)."""
        potential = MagicMock()
        structure = Atoms("Si")

        with patch("mlip_autopipec.validation.metrics.PhononValidator._calculate_band_structure") as mock_calc:
            mock_calc.return_value = -2.0  # Imaginary frequency

            result = PhononValidator.run(potential, structure)
            assert result.passed is False
            assert result.score == -2.0
