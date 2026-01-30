from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def test_uat_cycle_02_missing_executable(tmp_path: Path) -> None:
    """UAT-C02-02: Handle missing executable."""
    config_path = tmp_path / "config.yaml"
    yaml_content = """
    project_name: "UAT_Project"
    potential:
      elements: ["Si"]
      cutoff: 3.0
    lammps:
      command: "/path/to/nothing"
      timeout: 10
    structure_gen:
        element: "Si"
    """
    config_path.write_text(yaml_content)

    # Mock shutil.which to return None
    with patch("shutil.which", return_value=None):
        result = runner.invoke(app, ["run-cycle-02", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "Executable" in result.stdout or "not found" in result.stdout


@patch("mlip_autopipec.infrastructure.io.run_subprocess")
@patch("mlip_autopipec.infrastructure.io.read_lammps_dump")
@patch("shutil.which")
def test_uat_cycle_02_success(mock_which, mock_read, mock_run, tmp_path: Path) -> None:
    """UAT-C02-01: One-Shot MD Run Success."""
    config_path = tmp_path / "config.yaml"
    yaml_content = """
    project_name: "UAT_Project_Success"
    potential:
      elements: ["Si"]
      cutoff: 3.0
    lammps:
      command: "lmp_mock"
    structure_gen:
        element: "Si"
    """
    config_path.write_text(yaml_content)

    # Mock executable existence
    mock_which.return_value = "/usr/bin/lmp_mock"

    # Mock subprocess success
    mock_run.return_value = ("Simulation successful", "")

    # Mock read dump (return list of Atoms)
    import ase
    mock_atoms = ase.Atoms("Si2", positions=[[0,0,0], [1,1,1]], cell=[5,5,5], pbc=True)
    mock_read.return_value = [mock_atoms]

    with runner.isolated_filesystem(temp_dir=tmp_path):
         # Write config again in isolated dir
        (Path.cwd() / "config.yaml").write_text(yaml_content)
        result = runner.invoke(app, ["run-cycle-02", "--config", "config.yaml"])

    if result.exit_code != 0:
        print(result.stdout)
    assert result.exit_code == 0
    assert "Simulation Completed" in result.stdout
