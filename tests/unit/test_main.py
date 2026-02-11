from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_cli_run_success(valid_config_dict: dict[str, Any], tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    work_dir = tmp_path / "work"
    valid_config_dict["orchestrator"]["work_dir"] = str(work_dir)

    with config_path.open("w") as f:
        yaml.dump(valid_config_dict, f)

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 0
    assert "Workflow Completed" in result.stdout

def test_cli_missing_file() -> None:
    result = runner.invoke(app, ["run", "non_existent.yaml"])

    assert result.exit_code == 1
    # Check output which captures both stdout and stderr
    assert "Config file not found" in result.output
