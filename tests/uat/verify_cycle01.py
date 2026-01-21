from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.models import DFTResult
from mlip_autopipec.dft.qe_runner import QERunner

runner = CliRunner()

def test_scenario_1_1_config_validation(tmp_path: Path) -> None:
    # Valid
    config_file = tmp_path / "valid.yaml"
    config_file.write_text("""
global:
  project_name: "UAT"
  database_path: "uat.db"
dft:
  command: "pw.x"
  pseudopotential_dir: "."
  ecutwfc: 40
    """)
    result = runner.invoke(app, ["check-config", str(config_file)])
    assert result.exit_code == 0
    assert "OK" in result.stdout

    # Invalid
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("global: {}")
    result = runner.invoke(app, ["check-config", str(invalid_file)])
    assert result.exit_code == 1
    assert "Validation Error" in result.stdout

def test_scenario_1_2_static_calculation(tmp_path: Path, mocker: MagicMock) -> None:
    # Setup
    db_path = tmp_path / "uat.db"
    db = DatabaseManager(db_path)

    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Si.upf").touch()

    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30
    )
    qe_runner = QERunner(config)

    atoms = Atoms("Si2", positions=[[0,0,0], [1,1,1]], cell=[5,5,5], pbc=True)

    # Mock execution and parsing
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = MagicMock(returncode=0)

    # Need to mock stdout write to prevent file not found error?
    # QERunner._execute_calculation opens output_file in write mode.
    # It passes the file object to subprocess.run(stdout=...).
    # If subprocess.run is mocked, nothing is written to file.
    # But QERunner calls parse_pw_output(output_file).
    # If parse_pw_output reads the file, it will be empty!
    # Mock parse_pw_output
    expected_result = DFTResult(energy=-21.0, forces=np.zeros((2,3)), stress=np.zeros(6))
    mocker.patch("mlip_autopipec.dft.qe_runner.parse_pw_output", return_value=expected_result)

    # Execute
    result = qe_runner.run_static_calculation(atoms, tmp_path / "run_si")

    # Verify DB insertion
    calc = SinglePointCalculator(atoms, energy=result.energy, forces=result.forces, stress=result.stress)
    atoms.calc = calc

    db.add_calculation(atoms, {"config_type": "scf"})

    # Verify
    assert db.count() == 1
    row = db._connection.get(id=1)
    assert row.energy == -21.0

def test_scenario_1_3_magnetism(tmp_path: Path, mocker: MagicMock) -> None:
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.upf").touch()

    config = DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30
    )
    qe_runner = QERunner(config)

    atoms = Atoms("Fe", cell=[2.8, 2.8, 2.8], pbc=True)

    # Mock subprocess to intercept input file
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = MagicMock(returncode=0)
    mocker.patch("mlip_autopipec.dft.qe_runner.parse_pw_output",
                 return_value=DFTResult(energy=0, forces=np.zeros((1,3)), stress=np.zeros(6)))

    qe_runner.run_static_calculation(atoms, tmp_path / "run_fe")

    # Inspect input file
    input_file = tmp_path / "run_fe" / "pw.in"
    content = input_file.read_text()

    assert "nspin" in content
    assert "2" in content
    # starting_magnetization should be present for Fe (atom 1, or type 1)
    # ASE usually writes starting_magnetization(1) if type 1 has magmom.
    assert "starting_magnetization" in content
