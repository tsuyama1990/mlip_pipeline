from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()

def test_init_creates_template(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (Path("input.yaml")).exists()

        # Verify content
        with Path("input.yaml").open() as f:
            data = yaml.safe_load(f)
        assert "target_system" in data
        assert "dft" in data

def test_validate_valid_config(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create valid config
        runner.invoke(app, ["init"])

        # Override pseudopotential_dir to a valid path for validation
        with Path("input.yaml").open() as f:
            data = yaml.safe_load(f)

        data["dft"]["pseudopotential_dir"] = str(tmp_path)

        with Path("input.yaml").open("w") as f:
            yaml.dump(data, f)

        result = runner.invoke(app, ["validate", "input.yaml"])
        assert result.exit_code == 0
        assert "Validation Successful" in result.stdout

def test_validate_invalid_config(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("bad_config.yaml").open("w") as f:
            f.write("target_system: []\n") # Invalid type

        result = runner.invoke(app, ["validate", "bad_config.yaml"])
        assert result.exit_code == 1
        assert "Validation Error" in result.stdout

def test_db_init(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])

        # Fix path
        with Path("input.yaml").open() as f:
            data = yaml.safe_load(f)
        data["dft"]["pseudopotential_dir"] = str(tmp_path)
        with Path("input.yaml").open("w") as f:
            yaml.dump(data, f)

        result = runner.invoke(app, ["db", "init", "--config", "input.yaml"])
        assert result.exit_code == 0
        assert "Database initialized" in result.stdout
        assert Path("mlip.db").exists()
