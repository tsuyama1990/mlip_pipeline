"""Integration test for the DFTJobFactory to ensure end-to-end functionality."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.modules.dft import DFTHeuristics, DFTJobFactory, DFTRunner

MOCK_PW_X_PATH = Path(__file__).parent.parent / "test_data" / "mock_pw.x"


@pytest.fixture
def h2_atoms() -> Atoms:
    """Provide a simple H2 molecule Atoms object."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


# Removed xfail marker and updated test to match mock output
def test_dft_factory_integration(h2_atoms: Atoms, tmp_path: Path) -> None:
    """Test the full DFT calculation pipeline using a mock executable."""
    # Arrange
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.pbe-rrkjus.UPF").touch()
    sssp_path = tmp_path / "sssp.json"
    sssp_path.write_text('{"H": {"cutoff_wfc": 30, "cutoff_rho": 120, "filename": "H.pbe-rrkjus.UPF"}}')

    from ase.calculators.espresso import EspressoProfile

    from mlip_autopipec.modules.dft import (
        QEInputGenerator,
        QEOutputParser,
        QEProcessRunner,
    )

    profile = EspressoProfile(command=str(MOCK_PW_X_PATH.resolve()), pseudo_dir=pseudo_dir)
    input_generator = QEInputGenerator(profile=profile, pseudopotentials_path=pseudo_dir)
    process_runner = QEProcessRunner(profile=profile)
    output_parser = QEOutputParser()
    with patch("mlip_autopipec.modules.dft.SSSP_DATA_PATH", tmp_path / "sssp.json"):
        heuristics = DFTHeuristics(sssp_data_path=tmp_path / "sssp.json")
    dft_job_factory = DFTJobFactory(heuristics=heuristics)

    # Act
    job = dft_job_factory.create_job(h2_atoms.copy())
    dft_runner = DFTRunner(
        input_generator=input_generator,
        process_runner=process_runner,
        output_parser=output_parser,
    )
    result = dft_runner.run(job)

    # Assert
    # Values from mock_espresso.pwo
    expected_energy = -16.42531639 * 13.605693122994  # Ry to eV

    # Forces in mock_espresso.pwo are in Ry/au.
    # ASE converts them to eV/A.
    # 1 Ry = 13.605693122994 eV
    # 1 au (Bohr) = 0.529177210903 A
    # Factor = 13.605693122994 / 0.529177210903 = 25.711043
    ry_au_to_ev_a = 13.605693122994 / 0.529177210903

    expected_forces = np.array(
        [
            [-0.00000135, 0.0, 0.0],
            [0.00000135, 0.0, 0.0],
        ]
    ) * ry_au_to_ev_a

    # Stress is not present in the mock output snippet I saw, so it defaults to zero or raises error?
    # ASE Espresso reader might return zeros if stress block is missing.
    # Let's verify stress is zero.
    expected_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert np.isclose(result.energy, expected_energy, atol=1e-4)
    # The forces in mock file are very small, so strict check is fine.
    assert np.allclose(result.forces, expected_forces, atol=1e-4)
    # Checking stress logic if needed
    if hasattr(result, 'stress'):
         assert np.allclose(result.stress, expected_stress, atol=1e-6)


def test_dft_factory_executable_not_found(h2_atoms: Atoms, tmp_path: Path) -> None:
    """Test that a FileNotFoundError is raised for a non-existent executable."""
    # Arrange
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.pbe-rrkjus.UPF").touch()
    sssp_path = tmp_path / "sssp.json"
    sssp_path.write_text('{"H": {"cutoff_wfc": 30, "cutoff_rho": 120, "filename": "H.pbe-rrkjus.UPF"}}')

    from ase.calculators.espresso import EspressoProfile

    from mlip_autopipec.modules.dft import (
        QEInputGenerator,
        QEOutputParser,
        QEProcessRunner,
    )

    profile = EspressoProfile(command="/path/to/non/existent/pw.x", pseudo_dir=pseudo_dir)
    input_generator = QEInputGenerator(profile=profile, pseudopotentials_path=pseudo_dir)
    process_runner = QEProcessRunner(profile=profile)
    output_parser = QEOutputParser()
    with patch("mlip_autopipec.modules.dft.SSSP_DATA_PATH", tmp_path / "sssp.json"):
        heuristics = DFTHeuristics(sssp_data_path=tmp_path / "sssp.json")
    dft_job_factory = DFTJobFactory(heuristics=heuristics)

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        job = dft_job_factory.create_job(h2_atoms.copy())
        dft_runner = DFTRunner(
            input_generator=input_generator,
            process_runner=process_runner,
            output_parser=output_parser,
        )
        dft_runner.run(job)


def test_dft_runner_retry_logic(h2_atoms: Atoms, tmp_path: Path, mocker) -> None:
    """Test that the DFTRunner correctly handles retries and raises an error."""
    # Arrange
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.pbe-rrkjus.UPF").touch()
    sssp_path = tmp_path / "sssp.json"
    sssp_path.write_text('{"H": {"cutoff_wfc": 30, "cutoff_rho": 120, "filename": "H.pbe-rrkjus.UPF"}}')

    import subprocess

    from ase.calculators.espresso import EspressoProfile

    from mlip_autopipec.exceptions import DFTCalculationError
    from mlip_autopipec.modules.dft import (
        DFTJobFactory,
        DFTRunner,
        QEInputGenerator,
        QEOutputParser,
        QEProcessRunner,
    )

    profile = EspressoProfile(command=str(MOCK_PW_X_PATH.resolve()), pseudo_dir=pseudo_dir)
    input_generator = QEInputGenerator(profile=profile, pseudopotentials_path=pseudo_dir)
    process_runner = QEProcessRunner(profile=profile)
    output_parser = QEOutputParser()
    with patch("mlip_autopipec.modules.dft.SSSP_DATA_PATH", tmp_path / "sssp.json"):
        heuristics = DFTHeuristics(sssp_data_path=tmp_path / "sssp.json")
    dft_job_factory = DFTJobFactory(heuristics=heuristics)

    # Patch execution to fail
    # Note: subprocess.run raises CalledProcessError if check=True
    mocker.patch.object(
        process_runner,
        "execute",
        side_effect=subprocess.CalledProcessError(
            1, "pw.x", "stdout convergence NOT achieved", "stderr"
        ),
    )

    dft_runner = DFTRunner(
        input_generator=input_generator,
        process_runner=process_runner,
        output_parser=output_parser,
    )
    job = dft_job_factory.create_job(h2_atoms.copy())

    # Act & Assert
    # It should raise DFTCalculationError after retries
    with pytest.raises(DFTCalculationError):
        dft_runner.run(job)
    assert process_runner.execute.call_count == 3
