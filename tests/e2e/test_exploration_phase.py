from typer.testing import CliRunner
from mlip_autopipec.app import app
from unittest.mock import patch
import numpy as np

runner = CliRunner()

@patch("mlip_autopipec.cli.commands.io.run_subprocess")
@patch("mlip_autopipec.cli.commands.io.read_lammps_dump")
@patch("mlip_autopipec.cli.commands.io.write_lammps_data")
def test_run_cycle_02_command(mock_write, mock_read, mock_run, tmp_path):
    """Test the Cycle 02 run command."""
    # Setup mock return values
    mock_run.return_value = None

    # Mock reading dump to return a dummy structure
    from mlip_autopipec.domain_models.structure import Structure
    dummy_struct = Structure(
        symbols=["Si"]*8,
        positions=np.zeros((8, 3)),
        cell=np.eye(3)*5.43,
        pbc=(True, True, True)
    )
    mock_read.return_value = dummy_struct

    # Create a config file
    config_path = tmp_path / "config.yaml"
    config_content = """
    project_name: "Test"
    potential:
        elements: ["Si"]
        cutoff: 5.0
    structure_gen:
        element: "Si"
        crystal_structure: "diamond"
        lattice_constant: 5.43
    lammps:
        command: "echo"
    """
    config_path.write_text(config_content)

    # Run command
    result = runner.invoke(app, ["run-cycle-02", "--config", str(config_path)])

    # Assertions
    # Note: These will fail until implementation is complete
    assert result.exit_code == 0
    assert "Cycle 02 Completed" in result.stdout
