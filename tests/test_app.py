from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def test_app_run_valid(tmp_path: Path) -> None:
    # Create valid config
    config_data = {
        "project_name": "AppTest",
        "target_system": {"elements": ["Fe"], "composition": {"Fe": 1.0}},
        "resources": {"dft_code": "quantum_espresso", "parallel_cores": 2},
    }
    input_file = tmp_path / "input.yaml"
    with input_file.open("w") as f:
        yaml.dump(config_data, f)

    # Run in the temp directory so project is created there
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run", str(input_file)])
        assert result.exit_code == 0
        assert "System initialized" in result.stdout

        # Verify side effects
        project_dir = Path("AppTest")
        assert project_dir.exists()
        assert (project_dir / "AppTest.db").exists()
        assert (project_dir / "system.log").exists()


def test_app_run_invalid_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "nonexistent.yaml"])
    assert result.exit_code != 0
    # Typer (via Click) usually prints "does not exist" for Path(exists=True)
    # Note: Rich might be capturing stderr/stdout, but Click writes to stderr for errors
    assert "does not exist" in result.stdout or "does not exist" in result.stderr


def test_app_run_invalid_config(tmp_path: Path) -> None:
    config_data = {
        "project_name": "BadAppTest",
        "target_system": {
            "elements": ["Fe"],
            "composition": {"Fe": 0.5},  # Invalid sum
        },
        "resources": {"dft_code": "quantum_espresso", "parallel_cores": 2},
    }
    input_file = tmp_path / "bad_input.yaml"
    with input_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(input_file)])
    assert result.exit_code != 0
    # Output should contain validation error
    assert (
        "Composition must sum to 1.0" in str(result.exception)
        or "Composition must sum to 1.0" in result.stdout
    )
