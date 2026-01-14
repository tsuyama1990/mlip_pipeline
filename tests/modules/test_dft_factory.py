"""Integration tests for the QEProcessRunner module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.system import DFTParams, SystemConfig
from mlip_autopipec.modules.dft_factory import (
    DFTCalculationError,
    QEInputGenerator,
    QEProcessRunner,
)

# A canonical, complete Quantum Espresso output that ASE can parse
SAMPLE_QE_OUTPUT = """
     Program PWSCF v.6.5 starts on 10Jan2024 at 10:10:10
     bravais-lattice index     = 0
     lattice parameter (a_0)   = 18.897261  a.u.
     celldm(1)=   18.897261
     number of atoms/cell      = 1
     number of atomic types    = 1
     crystal axes: (cart. coord. in units of a_0)
               a(1) = (   0.529177   0.000000   0.000000 )
               a(2) = (   0.000000   0.529177   0.000000 )
               a(3) = (   0.000000   0.000000   0.529177 )
     site n.     atom                  positions (alat units)
         1           Ni          tau(   1) = (   0.0000000   0.0000000   0.0000000  )
!    total energy              =      -1.00000000 Ry
     Forces acting on atoms (Ry/au):

     atom    1 type  1   force =     0.100000000   0.200000000   0.300000000

     Total force =     0.374166     Total SCF correction =     0.000000
     total   stress  (Ry/bohr**3)     (kbar)     P=   -0.00
      0.00000000   0.00000000   0.00000000
      0.00000000   0.00000000   0.00000000
      0.00000000   0.00000000   0.00000000
     JOB DONE.
"""


@pytest.fixture
def sample_system_config() -> SystemConfig:
    """Provide a sample SystemConfig for a Nickel calculation."""
    dft_params = DFTParams(
        pseudopotentials={"Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"},
        system={
            "nat": 1,
            "ntyp": 1,
            "ecutwfc": 60.0,
            "nspin": 2,
        },
    )
    return SystemConfig(dft=dft_params, db_path="test.db")


@pytest.fixture
def sample_atoms() -> Atoms:
    """Provide a sample single-atom ASE Atoms object."""
    atoms = Atoms("Ni", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
    # Attach a calculator so that atoms.calc.results exists
    atoms.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
        atoms, energy=0.0, forces=[[0, 0, 0]]
    )
    return atoms


def test_qeinputgenerator_generate(
    sample_system_config: SystemConfig, sample_atoms: Atoms
) -> None:
    """Test the generation of the QE input file content."""
    generator = QEInputGenerator(sample_system_config)
    input_content = generator.generate(sample_atoms)

    assert "&CONTROL" in input_content
    assert "calculation = 'scf'" in input_content
    assert "&SYSTEM" in input_content
    assert "nat = 1" in input_content
    assert "nspin = 2" in input_content
    assert "ATOMIC_SPECIES" in input_content
    assert "Ni 1.0 Ni.pbe-n-rrkjus_psl.1.0.0.UPF" in input_content
    assert "ATOMIC_POSITIONS {angstrom}" in input_content
    assert "Ni 0.0 0.0 0.0" in input_content


@patch("mlip_autopipec.modules.dft_factory.QEInputGenerator.generate")
@patch("mlip_autopipec.modules.dft_factory.QEProcessRunner._execute_pw_x")
@patch("mlip_autopipec.modules.dft_factory.QEProcessRunner._parse_output")
def test_qeprocessrunner_run_orchestration(
    mock_parse: MagicMock,
    mock_execute: MagicMock,
    mock_generate: MagicMock,
    sample_system_config: SystemConfig,
    sample_atoms: Atoms,
) -> None:
    """Test that the run method correctly orchestrates its internal calls."""
    mock_generate.return_value = "dummy input file content"
    mock_parse.return_value = {"energy": -1.0, "forces": [[0, 0, 0]], "stress": [0] * 6}

    runner = QEProcessRunner(sample_system_config)
    result_atoms = runner.run(sample_atoms)

    mock_generate.assert_called_once_with(sample_atoms)
    mock_execute.assert_called_once()
    mock_parse.assert_called_once()
    assert result_atoms.calc.results["energy"] == -1.0


@patch("mlip_autopipec.modules.dft_factory.QEInputGenerator.generate")
@patch("mlip_autopipec.modules.dft_factory.QEProcessRunner._execute_pw_x")
@patch("mlip_autopipec.modules.dft_factory.QEProcessRunner._parse_output")
def test_qeprocessrunner_run_orchestration_failure(
    mock_parse: MagicMock,
    mock_execute: MagicMock,
    mock_generate: MagicMock,
    sample_system_config: SystemConfig,
    sample_atoms: Atoms,
) -> None:
    """Test that the run method correctly handles a failure in the execution step."""
    mock_generate.return_value = "dummy input file content"
    mock_execute.side_effect = DFTCalculationError("Execution failed")

    runner = QEProcessRunner(sample_system_config)
    with pytest.raises(DFTCalculationError, match="Execution failed"):
        runner.run(sample_atoms)

    mock_generate.assert_called_once_with(sample_atoms)
    mock_execute.assert_called_once()
    mock_parse.assert_not_called()


def test_parse_output_happy_path(
    sample_system_config: SystemConfig, tmp_path: Path
) -> None:
    """Test the _parse_output method with a valid QE output file."""
    output_path = tmp_path / "dft.out"
    output_path.write_text(SAMPLE_QE_OUTPUT)

    runner = QEProcessRunner(sample_system_config)
    results = runner._parse_output(output_path)

    assert "energy" in results
    assert results["energy"] == pytest.approx(-13.60569)
    assert "forces" in results
    assert results["forces"][0][0] == pytest.approx(0.1 * 13.60569 / 0.529177)


@patch("subprocess.run")
def test_qeprocessrunner_run_failure_path(
    mock_subprocess_run: MagicMock,
    sample_system_config: SystemConfig,
    sample_atoms: Atoms,
) -> None:
    """Test that a non-zero exit code from pw.x raises a DFTCalculationError."""
    # Mock the subprocess to simulate a failed run
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="pw.x", stderr="SCF failed to converge"
    )

    runner = QEProcessRunner(sample_system_config)
    with pytest.raises(DFTCalculationError, match="DFT calculation failed"):
        runner.run(sample_atoms)
