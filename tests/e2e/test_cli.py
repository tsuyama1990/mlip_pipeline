import pytest
import yaml
from pathlib import Path
from typer.testing import CliRunner

try:
    from mlip_autopipec.main import app
except ImportError:
    app = None # type: ignore

runner = CliRunner()

@pytest.mark.skipif(app is None, reason="CLI app not implemented yet")
def test_help() -> None:
    if app:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.stdout

@pytest.mark.skipif(app is None, reason="CLI app not implemented yet")
def test_run_command(tmp_path: Path) -> None:
    if app:
        config_file = tmp_path / "config.yaml"
        work_dir = tmp_path / "work"
        config_data = {
            "work_dir": str(work_dir),
            "max_cycles": 1,
            "random_seed": 42
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)
        result = runner.invoke(app, ["run", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "Active Learning Loop Finished" in result.stdout
        assert (work_dir / "mlip_pipeline.log").exists()

@pytest.mark.skipif(app is None, reason="CLI app not implemented yet")
def test_config_not_found(tmp_path: Path) -> None:
    if app:
        result = runner.invoke(app, ["run", "--config", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.stderr
