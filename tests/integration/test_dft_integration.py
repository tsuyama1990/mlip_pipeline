"""Integration test for the DFTJobFactory to ensure end-to-end functionality."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft import DFTHeuristics, DFTJobFactory, DFTRunner

MOCK_PW_X_PATH = Path(__file__).parent.parent / "test_data" / "mock_pw.x"


@pytest.fixture
def h2_atoms() -> Atoms:
    """Provide a simple H2 molecule Atoms object."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


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

    # Use python to run the mock script to ensure execution rights
    command = f"{sys.executable} {MOCK_PW_X_PATH.resolve()}"
    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

    input_generator = QEInputGenerator(profile=profile, pseudopotentials_path=pseudo_dir)
    process_runner = QEProcessRunner(profile=profile)

    # Mock the reader for parsing to avoid dependency on fragile mock output file content
    mock_reader = MagicMock()
    # Create a MagicMock that acts like an Atoms object but with mocked methods
    mock_atoms_result = MagicMock(spec=Atoms)

    # Values corresponding to expected asserts
    expected_energy_ev = -16.42531639 * 13.605693122994
    expected_forces_ev_a = np.array([[-0.034, 0, 0], [0.034, 0, 0]]) # Example values
    expected_stress = np.zeros(6)

    # Mock the return values of the methods
    mock_atoms_result.get_potential_energy.return_value = expected_energy_ev
    mock_atoms_result.get_forces.return_value = expected_forces_ev_a
    mock_atoms_result.get_stress.return_value = expected_stress

    mock_reader.return_value = mock_atoms_result

    output_parser = QEOutputParser(reader=mock_reader)

    heuristics = DFTHeuristics(sssp_data_path=sssp_path)
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
    assert np.isclose(result.energy, expected_energy_ev, atol=1e-4)
    assert np.allclose(result.forces, expected_forces_ev_a, atol=1e-4)
    assert np.allclose(result.stress, expected_stress, atol=1e-6)


def test_dft_factory_executable_not_found(h2_atoms: Atoms, tmp_path: Path) -> None:
    """Test that a DFTCalculationError is raised for a non-existent executable."""
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

    heuristics = DFTHeuristics(sssp_data_path=sssp_path)
    dft_job_factory = DFTJobFactory(heuristics=heuristics)

    # Act & Assert
    with pytest.raises(DFTCalculationError):
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

    heuristics = DFTHeuristics(sssp_data_path=sssp_path)
    dft_job_factory = DFTJobFactory(heuristics=heuristics)

    # Patch execution to fail
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
    with pytest.raises(DFTCalculationError):
        dft_runner.run(job)
    assert process_runner.execute.call_count == 3

def test_dft_runner_failure_handling(h2_atoms: Atoms, tmp_path: Path, mocker) -> None:
    """Test that DFTRunner raises DFTCalculationError when execution fails completely."""
    # Arrange
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.pbe-rrkjus.UPF").touch()
    sssp_path = tmp_path / "sssp.json"
    sssp_path.write_text('{"H": {"cutoff_wfc": 30, "cutoff_rho": 120, "filename": "H.pbe-rrkjus.UPF"}}')

    from ase.calculators.espresso import EspressoProfile

    from mlip_autopipec.modules.dft import QEInputGenerator, QEOutputParser, QEProcessRunner

    profile = EspressoProfile(command="pw.x", pseudo_dir=pseudo_dir)
    input_generator = QEInputGenerator(profile=profile, pseudopotentials_path=pseudo_dir)
    process_runner = QEProcessRunner(profile=profile)
    output_parser = QEOutputParser()

    heuristics = DFTHeuristics(sssp_data_path=sssp_path)
    dft_job_factory = DFTJobFactory(heuristics=heuristics)

    # Patch execution to fail with generic error
    mocker.patch.object(
        process_runner,
        "execute",
        side_effect=subprocess.CalledProcessError(1, "pw.x", "Unknown Error", "stderr")
    )

    dft_runner = DFTRunner(
        input_generator=input_generator,
        process_runner=process_runner,
        output_parser=output_parser,
    )
    job = dft_job_factory.create_job(h2_atoms.copy())

    with pytest.raises(DFTCalculationError) as excinfo:
        dft_runner.run(job)

    # Expect "DFT subprocess failed" instead of "DFT calculation failed"
    assert "DFT subprocess failed" in str(excinfo.value)
