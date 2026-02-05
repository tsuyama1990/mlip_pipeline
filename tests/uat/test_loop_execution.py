from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_mock_loop(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Create config file
    config_data = {
        "execution_mode": "mock",
        "max_cycles": 2,  # Reduce to 2 for faster test
        "exploration": {"strategy_name": "random", "max_structures": 2},
        "dft": {"calculator": "espresso"},
        "training": {"fitting_code": "pacemaker"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Change working directory to tmp_path so mock files are created there
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run", str(config_file)])

        # Check stdout
        assert result.exit_code == 0
        assert "Starting Cycle 1" in result.stdout
        assert "Workflow completed successfully" in result.stdout

        # Check file creation (Integration Verification)
        # We expect mock_structure_*.xyz files and mock_potential.yace
        xyz_files = list(Path().glob("mock_structure_*.xyz"))
        assert len(xyz_files) > 0, "No mock structure XYZ files were created"

        potential_file = Path("mock_potential.yace")
        assert potential_file.exists(), "Mock potential file was not created"
        assert potential_file.read_text() == "mock potential content"
