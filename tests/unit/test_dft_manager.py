from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.oracle.dft_manager import DFTManager, _process_structure_wrapper


class TestDFTManager:
    def test_compute_success(self) -> None:
        # Create a real structure
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structure = Structure(atoms=atoms, provenance="test", label_status="unlabeled")
        structures = [structure]

        config = OracleConfig(type=OracleType.DFT, n_workers=1)
        manager = DFTManager(config)

        labeled_structure = Structure(
            atoms=atoms,
            provenance="test",
            label_status="labeled",
            energy=-10.0,
            forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            stress=[0.0] * 6
        )

        with (
            patch(
                "mlip_autopipec.oracle.dft_manager.ProcessPoolExecutor",
                side_effect=ThreadPoolExecutor,
            ),
            patch(
                "mlip_autopipec.oracle.dft_manager._process_structure_wrapper",
                return_value=labeled_structure,
            ) as mock_wrapper,
        ):
            results = list(manager.compute(structures))
            assert len(results) == 1
            assert results[0].label_status == "labeled"
            assert results[0].energy == -10.0
            mock_wrapper.assert_called_once()

    def test_compute_failure(self) -> None:
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structure = Structure(atoms=atoms, provenance="test", label_status="unlabeled")
        structures = [structure]

        config = OracleConfig(type=OracleType.DFT, n_workers=1)
        manager = DFTManager(config)

        def mock_process_structure_fail(structure: Structure, config: OracleConfig) -> Structure:
            msg = "Simulated failure"
            raise RuntimeError(msg)

        with (
            patch(
                "mlip_autopipec.oracle.dft_manager.ProcessPoolExecutor",
                side_effect=ThreadPoolExecutor,
            ),
            patch(
                "mlip_autopipec.oracle.dft_manager._process_structure_wrapper",
                side_effect=mock_process_structure_fail,
            ),
        ):
            results = list(manager.compute(structures))
            assert len(results) == 1
            assert results[0].label_status == "failed"
            assert results[0].energy is None

    def test_process_structure_wrapper_logic(self) -> None:
        """
        Test the standalone wrapper function to ensure it attaches/detaches calculator correctly.
        """
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structure = Structure(atoms=atoms, provenance="test", label_status="unlabeled")
        config = OracleConfig(type=OracleType.DFT, command="mpirun -np 4 pw.x")

        # Create a mock calculator that returns energy/forces
        mock_calc = MagicMock(spec=Calculator)
        mock_calc.get_potential_energy.return_value = -15.0

        # Mocking get_forces needs to return numpy array if real ase calls are made,
        import numpy as np
        mock_calc.get_forces.return_value = np.array([[0.1, 0.1, 0.1]] * 2)
        mock_calc.get_stress.return_value = np.array([0.0] * 6)

        # Mock QEDriver to return our mock calculator
        with (
            patch("mlip_autopipec.oracle.dft_manager.QEDriver") as MockDriver,
            patch("mlip_autopipec.oracle.dft_manager.run_with_healing") as mock_healing,
        ):
            mock_driver_instance = MockDriver.return_value
            mock_driver_instance.get_calculator.return_value = mock_calc

            result_structure = _process_structure_wrapper(structure, config)

            # Verification
            assert result_structure.label_status == "labeled"
            assert result_structure.energy == -15.0
            # Ensure the returned structure's atoms does NOT have a calculator attached
            assert result_structure.atoms.calc is None

            # Verify QEDriver was used
            MockDriver.assert_called_once()
            mock_healing.assert_called_once()
