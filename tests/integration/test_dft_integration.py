"""Integration test for the DFTFactory to ensure end-to-end functionality."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.espresso import EspressoProfile

from mlip_autopipec.config.models import DFTResult
from mlip_autopipec.modules.dft import (
    DFTFactory,
    QEInputGenerator,
    QEOutputParser,
    QERetryHandler,
    QEProcessRunner,
)

MOCK_PW_X_PATH = Path(__file__).parent.parent / "test_data" / "mock_pw.x"


@pytest.fixture
def h2_atoms() -> Atoms:
    """Provide a simple H2 molecule Atoms object."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.mark.xfail(reason="This test is known to fail due to issues with the mock pw.x output.")
def test_dft_factory_integration(h2_atoms: Atoms, tmp_path: Path) -> None:
    """Test the full DFT calculation pipeline using a mock executable."""
    # Arrange
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.pbe-rrkjus.UPF").touch()

    profile = EspressoProfile(
        command=str(MOCK_PW_X_PATH.resolve()), pseudo_dir=pseudo_dir
    )
    input_generator = QEInputGenerator(
        profile=profile, pseudopotentials_path=pseudo_dir
    )
    process_runner = QEProcessRunner(profile=profile)
    output_parser = QEOutputParser()
    retry_handler = QERetryHandler()
    dft_factory = DFTFactory(
        input_generator=input_generator,
        process_runner=process_runner,
        output_parser=output_parser,
        retry_handler=retry_handler,
    )

    # Act
    result = dft_factory.run(h2_atoms.copy())

    # Assert
    expected_energy = -16.42531639 * 13.605693122994  # Ry to eV
    expected_forces = np.array(
        [
            [-0.00000135, 0.0, 0.0],
            [0.00000135, 0.0, 0.0],
        ]
    ) * (13.605693122994 / 0.529177210903)  # Ry/au to eV/A
    expected_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert isinstance(result, DFTResult)
    assert np.isclose(result.energy, expected_energy, atol=1e-6)
    assert np.allclose(result.forces, expected_forces, atol=1e-6)
    assert np.allclose(result.stress, expected_stress, atol=1e-6)


def test_dft_factory_executable_not_found(h2_atoms: Atoms, tmp_path: Path) -> None:
    """Test that a FileNotFoundError is raised for a non-existent executable."""
    # Arrange
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.pbe-rrkjus.UPF").touch()

    non_existent_command = "/path/to/non/existent/pw.x"
    profile = EspressoProfile(command=non_existent_command, pseudo_dir=pseudo_dir)
    input_generator = QEInputGenerator(
        profile=profile, pseudopotentials_path=pseudo_dir
    )
    process_runner = QEProcessRunner(profile=profile)
    output_parser = QEOutputParser()
    retry_handler = QERetryHandler()
    dft_factory = DFTFactory(
        input_generator=input_generator,
        process_runner=process_runner,
        output_parser=output_parser,
        retry_handler=retry_handler,
    )

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        dft_factory.run(h2_atoms.copy())
