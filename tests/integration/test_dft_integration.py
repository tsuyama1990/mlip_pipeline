# ruff: noqa: S101
"""
Integration test for the DFTFactory to ensure end-to-end functionality.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config_schemas import (
    DFTConfig,
    DFTExecutable,
    DFTInput,
    SystemConfig,
)
from mlip_autopipec.modules.dft.factory import DFTFactory

# Define the path to the mock executable
# This path is relative to the root of the project where pytest is run
MOCK_PW_X_PATH = Path(__file__).parent.parent / "test_data" / "mock_pw.x"


@pytest.fixture
def system_config() -> SystemConfig:
    """
    Provides a SystemConfig instance pointing to the mock pw.x executable.
    """
    return SystemConfig(
        dft=DFTConfig(
            executable=DFTExecutable(command=str(MOCK_PW_X_PATH.resolve())),
            input=DFTInput(
                pseudopotentials={"H": "H.pbe-rrkjus.UPF"},
                control={},
                system={},
                electrons={},
            ),
        )
    )


@pytest.fixture
def h2_atoms() -> Atoms:
    """
    Provides a simple H2 molecule Atoms object.
    """
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


def test_dft_factory_integration(
    system_config: SystemConfig, h2_atoms: Atoms, tmp_path: Path
):
    """
    Tests the full DFT calculation pipeline using a mock executable.

    This test verifies that:
    1. A temporary directory is created for the calculation.
    2. The input file is correctly generated and written.
    3. The mock executable is called successfully.
    4. The output file is parsed correctly.
    5. The results (energy, forces, stress) are attached to the Atoms object.
    """
    # Arrange
    # The DFTFactory is instantiated with the configuration pointing to our mock
    dft_factory = DFTFactory(config=system_config.dft, base_work_dir=tmp_path)

    # Act
    # Run the DFT calculation, which will use the mock pw.x
    calculated_atoms = dft_factory.run(h2_atoms.copy())

    # Assert
    # 1. Check that a calculator has been attached
    assert calculated_atoms.calc is not None
    assert isinstance(calculated_atoms.calc, SinglePointCalculator)

    # 2. Check the parsed results against expected values from the mock output
    expected_energy = -16.42531639 * 13.605693122994  # Ry to eV
    expected_forces = np.array(
        [
            [-0.00000135, -0.00000000, 0.00000000],
            [0.00000135, 0.00000000, 0.00000000],
        ]
    ) * (
        13.605693122994 / 0.529177210903
    )  # Ry/au to eV/A

    # The ASE parser returns stress as a 3x3 matrix in GPa.
    # The QE output is in kbar, so we convert (1 kbar = 0.1 GPa).
    expected_stress_matrix_gpa = (
        np.array(
            [
                [-0.00, 0.00, 0.00],
                [0.00, -0.00, 0.00],
                [0.00, 0.00, -0.00],
            ]
        )
        * 0.1
    )

    # Retrieve results from the calculator
    results = calculated_atoms.calc.results
    assert "energy" in results
    assert "forces" in results
    assert "stress" in results

    assert np.isclose(results["energy"], expected_energy, atol=1e-6)
    assert np.allclose(results["forces"], expected_forces, atol=1e-6)
    assert np.allclose(results["stress"], expected_stress_matrix_gpa, atol=1e-6)

    # 3. Verify that the temporary calculation directory was created and then cleaned up
    # The specific directory name is unknown, but the base_work_dir should be empty
    # if cleanup is successful.
    assert not any(
        d.is_dir() for d in tmp_path.iterdir()
    ), "Temporary directory was not cleaned up."
