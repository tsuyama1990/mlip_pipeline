from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_mock_loop(tmp_path: Path) -> None:
    # Create config file
    config_data = {
        "execution_mode": "mock",
        "work_dir": str(tmp_path / "work"),
        "max_cycles": 2,  # Reduce to 2 for faster test
        "exploration": {"strategy_name": "random", "max_structures": 2},
        "dft": {"calculator": "espresso"},
        "training": {"fitting_code": "pacemaker"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Invoke CLI (no need for isolated_filesystem if we pass absolute work_dir, but safer to use it)
    # Actually, config.work_dir is used.
    # Let's run it.
    result = runner.invoke(app, ["run", str(config_file)])

    # Check stdout
    assert result.exit_code == 0
    assert "Starting Cycle 1" in result.stdout
    assert "Workflow completed successfully" in result.stdout

    # Check file creation (Integration Verification)
    work_dir = tmp_path / "work"
    assert work_dir.exists()

    # NOTE: MockExplorer no longer writes XYZ files to disk by default (Scalability Fix).
    # So we should NOT see .xyz files unless explicitly saved.
    xyz_files = list(work_dir.glob("mock_structure_*.xyz"))
    assert len(xyz_files) == 0, "Mock structure XYZ files should not be created on disk"

    potential_file = work_dir / "mock_potential.yace"
    assert potential_file.exists(), "Mock potential file was not created"
    assert potential_file.read_text() == "mock potential content"
