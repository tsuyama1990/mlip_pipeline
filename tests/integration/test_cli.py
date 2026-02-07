from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_init_command(tmp_path: Path) -> None:
    # Use tmp_path to isolate execution
    # We pass the path explicitly because runner.isolated_filesystem
    # might conflict with how pytest handles tmp_path or just be simpler to pass arg.
    config_path = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--path", str(config_path)])
    assert result.exit_code == 0
    assert config_path.exists()

    with config_path.open() as f:
        data = yaml.safe_load(f)
        assert data["project_name"] == "mlip_project_01"

def test_run_command(tmp_path: Path) -> None:
    # 1. Create config
    config_path = tmp_path / "config.yaml"
    result_init = runner.invoke(app, ["init", "--path", str(config_path)])
    assert result_init.exit_code == 0

    # 2. Modify config to ensure it uses tmp_path for workdir
    with config_path.open() as f:
        data = yaml.safe_load(f)

    workdir = tmp_path / "mlip_run"
    data["workdir"] = str(workdir)

    with config_path.open("w") as f:
        yaml.dump(data, f)

    # 3. Run
    result_run = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result_run.exit_code == 0

    # Check artifacts
    assert workdir.exists()
    assert (workdir / "potential.yace").exists()
