from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pydantic import ValidationError

from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.dft.qe_runner import QERunner
from mlip_autopipec.dft.utils import is_magnetic


def test_uat_1_1_config_validation(tmp_path):
    """
    Scenario 1.1: System Initialization & Config Validation
    """
    # Valid
    dft_config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotential_dir=tmp_path
    )
    assert dft_config.mixing_beta == 0.7

    # Invalid
    with pytest.raises(ValidationError):
        DFTConfig(
            code="quantum_espresso",
            command="pw.x",
            pseudopotential_dir=tmp_path,
            mixing_beta=1.5
        )

def test_uat_1_2_static_calculation_flow(tmp_path, mocker):
    """
    Scenario 1.2: Static DFT Calculation (Happy Path)
    """
    # Setup
    db_path = tmp_path / "uat.db"
    db = DatabaseManager(db_path)
    db.initialize()

    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Si.upf").touch()

    config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotential_dir=pseudo_dir
    )

    runner = QERunner(config)
    atoms = Atoms("Si", cell=[5,5,5], pbc=True)

    # Mock subprocess and parser to simulate successful run
    mock_run = mocker.patch("subprocess.run")
    mock_parser = mocker.patch("mlip_autopipec.dft.qe_runner.parse_pw_output")

    from decimal import Decimal

    from mlip_autopipec.data_models.dft_models import DFTResult
    mock_result = DFTResult(
        uid="uat_test", energy=Decimal("-21.0"), forces=[[0.0, 0.0, 0.0]], stress=[[0.0]*3]*3,
        succeeded=True, wall_time=10.0, parameters={"k_points": [1,1,1]}, final_mixing_beta=0.7
    )
    mock_parser.return_value = mock_result

    # Execute
    result = runner.run_static_calculation(atoms, tmp_path / "run")

    # Verification of Generated Input (Audit Request)
    input_file = tmp_path / "run" / "pw.in"
    assert input_file.exists()
    content = input_file.read_text()
    assert "calculation" in content
    assert "'scf'" in content
    assert "ATOMIC_POSITIONS" in content
    assert "Si" in content
    assert "CELL_PARAMETERS" in content

    # Store
    # Attach results to atoms for storage
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(result.energy), # ASE expects float
        forces=result.forces,
        stress=result.stress
    )

    # Add to DB
    db.add_calculation(atoms, {"calculation_type": "scf", "uid": result.uid})

    # Verify
    assert db.count() == 1
    row = db.get_atoms()[0]
    # Check if data is stored
    assert abs(row.get_potential_energy() - (-21.0)) < 1e-5

def test_uat_1_3_magnetism(tmp_path, mocker):
    """
    Scenario 1.3: Magnetism Auto-Detection
    """
    atoms = Atoms("Fe", cell=[2,2,2], pbc=True)
    assert is_magnetic(atoms)

    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.upf").touch()

    config = DFTConfig(
        code="quantum_espresso",
        command="pw.x",
        pseudopotential_dir=pseudo_dir
    )

    runner = QERunner(config)

    # Intercept write_pw_input to check parameters
    mock_write = mocker.patch("mlip_autopipec.dft.qe_runner.write_pw_input")
    mock_write.return_value = "fake input"

    # Mock execute/parse/clean to avoid errors
    mocker.patch("subprocess.run")
    mocker.patch("mlip_autopipec.dft.qe_runner.parse_pw_output", return_value=mocker.Mock())
    mocker.patch("shutil.rmtree")
    Path.unlink = mocker.Mock()

    runner.run_static_calculation(atoms, tmp_path / "run")

    # Check if nspin=2 was passed to write_pw_input
    call_args = mock_write.call_args
    params = call_args[0][1]
    assert params["system"]["nspin"] == 2
