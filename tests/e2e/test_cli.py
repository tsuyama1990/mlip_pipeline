import yaml
from typer.testing import CliRunner

from main import app

runner = CliRunner()

def test_run_command(tmp_path):
    config_data = {
        "work_dir": str(tmp_path / "work"),
        "max_cycles": 1,
        "random_seed": 42
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["--config", str(config_file)])
    assert result.exit_code == 0
    assert "Initializing Orchestrator..." in result.stdout
    # Check if work dir created
    assert (tmp_path / "work").exists()

def test_run_missing_config():
    result = runner.invoke(app, ["--config", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stdout

def test_run_invalid_yaml(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("key: value: broken")
    result = runner.invoke(app, ["--config", str(config_file)])
    assert result.exit_code == 1
    assert "Error parsing config file" in result.stdout

def test_run_invalid_schema(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    # Missing required fields
    config_file.write_text("random_seed: 42")
    result = runner.invoke(app, ["--config", str(config_file)])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.stdout
