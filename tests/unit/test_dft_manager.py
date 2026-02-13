from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from ase import Atoms

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.oracle.dft_manager import DFTManager


# Mock the worker function wrapper to avoid pickling issues with MagicMock in ProcessPoolExecutor
def mock_process_structure(structure: Structure, config: OracleConfig) -> Structure:
    atoms = structure.ase_atoms
    # Simulate calculation
    energy = -10.0
    forces = [[0.0, 0.0, 0.0]] * len(atoms)
    stress = [0.0] * 6

    return Structure(
        atoms=atoms,
        provenance=structure.provenance,
        label_status="labeled",
        energy=energy,
        forces=forces,
        stress=stress,
    )


def mock_process_structure_fail(structure: Structure, config: OracleConfig) -> Structure:
    msg = "Simulated failure"
    raise RuntimeError(msg)


class TestDFTManager:
    def test_compute_success(self) -> None:
        config = OracleConfig(type=OracleType.DFT, n_workers=2)
        manager = DFTManager(config)

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structures = [
            Structure(atoms=atoms, provenance="test", label_status="unlabeled") for _ in range(4)
        ]

        # Patch ProcessPoolExecutor to use ThreadPoolExecutor so mocks work
        with (
            patch(
                "mlip_autopipec.oracle.dft_manager.ProcessPoolExecutor",
                side_effect=ThreadPoolExecutor,
            ),
            patch(
                "mlip_autopipec.oracle.dft_manager._process_structure_wrapper",
                side_effect=mock_process_structure,
            ),
        ):
            # Consume iterator using deque to avoid building full list
            results_iter = manager.compute(structures)

            # Verify results one by one or count
            count = 0
            for res in results_iter:
                assert res.label_status == "labeled"
                assert res.energy == -10.0
                count += 1
            assert count == 4

    def test_compute_failure(self) -> None:
        config = OracleConfig(type=OracleType.DFT, n_workers=2)
        manager = DFTManager(config)

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        structures = [
            Structure(atoms=atoms, provenance="test", label_status="unlabeled") for _ in range(4)
        ]

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
            # Consume iterator
            results_iter = manager.compute(structures)

            count = 0
            for res in results_iter:
                assert res.label_status == "failed"
                assert res.energy is None
                count += 1
            assert count == 4
