import yaml
from pathlib import Path
from typer.testing import CliRunner
from mlip_autopipec.main import app

runner = CliRunner()

def test_run_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "valid_config.yaml"
    config_data = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0
    assert "Configuration loaded successfully" in result.stdout

def test_run_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_config.yaml"
    config_data = {
        "project_name": "TestProject",
        # Missing dft
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 1
    assert "Configuration validation failed" in result.stdout
    assert "dft" in result.stdout
