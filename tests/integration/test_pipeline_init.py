
import pytest
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()

@pytest.fixture
def input_yaml(tmp_path):
    data = {
        "project_name": "IntegrationTest",
        "target_system": {
            "elements": ["Fe"],
            "composition": {"Fe": 1.0},
            "crystal_structure": "bcc"
        },
        "simulation_goal": {
            "type": "elastic"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    p = tmp_path / "input.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p

def test_pipeline_init(input_yaml, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", str(input_yaml)])

    assert result.exit_code == 0, f"Stdout: {result.stdout}"
    assert "System initialized" in result.stdout

    # Check artifacts
    project_dir = tmp_path / "IntegrationTest"
    assert project_dir.exists()
    # Factory uses project_name.db
    assert (project_dir / "IntegrationTest.db").exists()
    assert (project_dir / "system.log").exists()

def test_pipeline_init_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Invalid YAML
    p = tmp_path / "bad.yaml"
    p.write_text("invalid: content")

    result = runner.invoke(app, ["init", str(p)])

    assert result.exit_code == 1
    assert "FAILURE" in result.stdout
