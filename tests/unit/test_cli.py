from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_cli_run_success(tmp_path: Path) -> None:
    config_data = {
        "work_dir": str(tmp_path),
        "logging_level": "DEBUG",
        "max_cycles": 1,
        "exploration": {"max_structures": 1}
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Pipeline completed successfully" in result.stdout
    assert (tmp_path / "mlip.log").exists()

def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stderr

def test_cli_run_invalid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    with config_file.open("w") as f:
        f.write("invalid: : yaml")

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    # Could vary depending on parser error
    assert "Pipeline failed" in result.stderr
