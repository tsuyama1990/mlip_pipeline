import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch
import mlip_autopipec.oracle.dft_manager

from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.oracle.dft_manager import DFTManager


# Top-level worker function for testing pickling support
def _test_worker_function(structure: Structure, config: OracleConfig) -> Structure:
    # Mimic the real worker logic but return a dummy result
    atoms = structure.ase_atoms
    # Ensure no calculator is attached to return value (simulating detached)
    atoms.calc = None
    return Structure(
        atoms=atoms,
        provenance="test",
        label_status="labeled",
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0]] * len(atoms),
        stress=[0.0] * 6
    )

def _test_worker_function_fail(structure: Structure, config: OracleConfig) -> Structure:
    msg = "Simulated failure"
    raise RuntimeError(msg)


class TestDFTManager:
    def test_compute_success(self) -> None:
        # Create a real structure
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structure = Structure(atoms=atoms, provenance="test", label_status="unlabeled")
        structures = [structure]

        config = OracleConfig(type=OracleType.DFT, n_workers=1)
        manager = DFTManager(config)

        # Direct monkeypatching to avoid MagicMock pickling issues with multiprocessing
        original_wrapper = mlip_autopipec.oracle.dft_manager._process_structure_wrapper
        mlip_autopipec.oracle.dft_manager._process_structure_wrapper = _test_worker_function

        try:
             # We use the REAL ProcessPoolExecutor here.
             results_iter = manager.compute(structures)
             result = next(results_iter)

             assert result.label_status == "labeled"
             assert result.energy == -10.0

             # Ensure iterator is exhausted
             try:
                 next(results_iter)
             except StopIteration:
                 pass
             else:
                 msg = "Iterator should be empty"
                 raise AssertionError(msg)
        finally:
            mlip_autopipec.oracle.dft_manager._process_structure_wrapper = original_wrapper

    def test_compute_failure(self) -> None:
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structure = Structure(atoms=atoms, provenance="test", label_status="unlabeled")
        structures = [structure]

        config = OracleConfig(type=OracleType.DFT, n_workers=1)
        manager = DFTManager(config)

        original_wrapper = mlip_autopipec.oracle.dft_manager._process_structure_wrapper
        mlip_autopipec.oracle.dft_manager._process_structure_wrapper = _test_worker_function_fail

        try:
            results_iter = manager.compute(structures)
            result = next(results_iter)

            assert result.label_status == "failed"
            assert result.energy is None
        finally:
            mlip_autopipec.oracle.dft_manager._process_structure_wrapper = original_wrapper

    def test_process_structure_wrapper_logic(self) -> None:
        """
        Test the standalone wrapper function to ensure it attaches/detaches calculator correctly.
        This runs in the main process, so we can use mocks.
        """
        # Need to import original wrapper since we might have patched it if tests run in parallel?
        # No, sequential. But safer to use imported one.
        from mlip_autopipec.oracle.dft_manager import _process_structure_wrapper

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
