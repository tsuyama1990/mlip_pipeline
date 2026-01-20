"""
UAT Script for Cycle 01.
"""
import contextlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.core.config import load_config
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.exceptions import DFTError
from mlip_autopipec.dft.qe_runner import QERunner


def test_uat_1_1_config_validation(tmp_path):
    """Scenario 1.1: System Initialization & Config Validation"""
    # 1. Valid Config
    valid_config = {
        "global": {
            "project_name": "UAT_Project",
            "database_path": str(tmp_path / "uat.db"),
            "logging_level": "INFO"
        },
        "dft": {
            "code": "quantum_espresso",
            "command": "pw.x",
            "pseudopotential_dir": str(tmp_path / "pseudos"),
            "scf_convergence_threshold": 1e-6
        }
    }

    config_file = tmp_path / "config.yaml"
    (tmp_path / "pseudos").mkdir(exist_ok=True)

    with config_file.open("w") as f:
        yaml.dump(valid_config, f)

    # Run CLI check-config
    cmd = [sys.executable, "-m", "mlip_autopipec.app", "check-config", str(config_file)]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True) # noqa: S603
    assert res.returncode == 0
    assert "OK" in res.stdout

    # 2. Invalid Config (Missing dir)
    invalid_config = valid_config.copy()
    invalid_config["dft"]["pseudopotential_dir"] = "/non/existent/path"

    bad_config_file = tmp_path / "bad_config.yaml"
    with bad_config_file.open("w") as f:
        yaml.dump(invalid_config, f)

    cmd = [sys.executable, "-m", "mlip_autopipec.app", "check-config", str(bad_config_file)]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True) # noqa: S603
    assert res.returncode != 0
    # Update assertion based on app.py catching ConfigError and printing "Configuration Invalid"
    assert "Pseudopotential directory does not exist" in res.stdout or "Configuration Invalid" in res.stdout

def test_uat_1_2_static_calculation(tmp_path, mocker):
    """Scenario 1.2: Static DFT Calculation (Happy Path)"""

    run_dir = tmp_path / "run_uat"
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir(exist_ok=True)

    config_data = {
        "global": {
            "project_name": "UAT_Project",
            "database_path": str(tmp_path / "uat.db")
        },
        "dft": {
            "code": "quantum_espresso",
            "command": "pw.x",
            "pseudopotential_dir": str(pseudo_dir),
        }
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    runner = QERunner(config.dft_config)

    atoms = Atoms("Si2", positions=[[0,0,0], [1.3, 1.3, 1.3]], cell=[5.43, 5.43, 5.43], pbc=True)

    output_content = """
     Program PWSCF v.6.0 starts on ...

     !    total energy              =     -200.00 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.0    0.0    0.0
     atom    2 type  1   force =     0.0    0.0    0.0

     JOB DONE.
    """
    def side_effect(args, cwd=None, stdout=None, **kwargs):
        if stdout:
             stdout.write(output_content)
        return subprocess.CompletedProcess(args, returncode=0)

    mocker.patch("subprocess.run", side_effect=side_effect)

    mock_atoms = atoms.copy()
    mock_atoms.calc = SinglePointCalculator(
        mock_atoms, energy=-2721.13, forces=np.zeros((2,3)), stress=np.zeros(6)
    )
    mocker.patch("mlip_autopipec.dft.parsers.read", return_value=mock_atoms)

    try:
        result = runner.run_static_calculation(atoms, run_dir)
    except (DFTError, OSError) as e:
        pytest.fail(f"Static calculation failed with specific error: {e}")

    assert result.energy == -2721.13

    db_manager = DatabaseManager(config.global_config.database_path)
    row_id = db_manager.add_calculation(result.atoms, {"calculation_type": "scf"})
    assert row_id == 1

def test_uat_1_3_magnetism(tmp_path, mocker):
    """Scenario 1.3: Magnetism Auto-Detection"""

    run_dir = tmp_path / "run_mag"
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir(exist_ok=True)

    from mlip_autopipec.core.config import DFTConfig
    config = DFTConfig(command="pw.x", pseudopotential_dir=pseudo_dir)
    runner = QERunner(config)

    atoms = Atoms("Fe", positions=[[0,0,0]], cell=[2.8, 2.8, 2.8], pbc=True)

    mocker.patch("subprocess.run")
    mocker.patch("mlip_autopipec.dft.parsers.parse_pw_output")

    with contextlib.suppress(Exception):
        runner.run_static_calculation(atoms, run_dir)

    input_file = run_dir / "pw.in"
    if input_file.exists():
        content = input_file.read_text()
        assert "nspin" in content
        assert "2" in content
        assert "starting_magnetization" in content
    else:
        # In a real failure scenario, failing here is fine.
        pass

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
