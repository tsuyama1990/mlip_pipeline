import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def setup_config(path: Path, lammps_cmd: str = "echo"):
    """Helper to create a valid config."""
    config_content = f"""
project_name: "UAT_Project"
potential:
  elements: ["Si"]
  cutoff: 5.0
  pair_style: "hybrid/overlay"
  npot: "FinnisSinclair"
  fs_parameters: [1.0, 1.0, 1.0, 0.5]
  ndensity: 2
structure_gen:
  strategy: "bulk"
  element: "Si"
  crystal_structure: "diamond"
  lattice_constant: 5.43
md:
  temperature: 300.0
  n_steps: 10
  timestep: 0.001
  ensemble: "NVT"
lammps:
  command: "{lammps_cmd}"
  timeout: 10
  use_mpi: false
"""
    (path / "config.yaml").write_text(config_content)


def create_mock_lammps_script(path: Path) -> Path:
    """Create a fake LAMMPS executable that writes a dummy dump file."""
    script_content = """#!/usr/bin/env python3
import sys
from pathlib import Path

# Create dump.lammpstrj
with open("dump.lammpstrj", "w") as f:
    f.write("ITEM: TIMESTEP\\n100\\n")
    f.write("ITEM: NUMBER OF ATOMS\\n1\\n")
    f.write("ITEM: BOX BOUNDS pp pp pp\\n0 10\\n0 10\\n0 10\\n")
    f.write("ITEM: ATOMS id type x y z c_pace_gamma\\n")
    f.write("1 1 0.0 0.0 0.0 0.0\\n")

# Create log.lammps
with open("log.lammps", "w") as f:
    f.write("LAMMPS (Mock)\\nStep Temp PotEng\\n100 300 -100\\nLoop time of 1.0\\n")

sys.exit(0)
"""
    script_path = path / "lmp_mock"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path


def test_uat_c02_01_one_shot_success(tmp_path):
    """
    Scenario 2.1: The "One-Shot" MD Run
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        td_path = Path(td)
        mock_exe = create_mock_lammps_script(td_path)

        # Use absolute path to mock executable
        setup_config(td_path, lammps_cmd=str(mock_exe))

        # We rely on real subprocess execution now, but pointing to our mock script
        # validation regex allows paths

        result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

        if result.exit_code != 0:
            print(result.stdout)
            print(result.stderr)

        assert result.exit_code == 0
        assert "Simulation Completed: Status COMPLETED" in result.stdout


def test_uat_c02_02_missing_executable(tmp_path):
    """
    Scenario 2.2: Missing Executable Handling
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td), lammps_cmd="/path/to/nothing")

        # Here we don't mock, we want it to fail naturally finding the executable
        # BUT shutil.which might check existence.
        # If LammpsRunner checks existence before running, it should fail gracefully.

        result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

        # It should exit with error
        assert result.exit_code != 0
        assert "Executable" in result.stdout and "not found" in result.stdout


if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))
