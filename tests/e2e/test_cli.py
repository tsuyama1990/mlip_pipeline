from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def test_cli_init(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (Path.cwd() / "input.yaml").exists()

        # Verify content
        with open("input.yaml") as f:
            content = f.read()
            assert "target_system:" in content


def test_cli_check_config_success(tmp_path):
    # Create dummy UPF dir
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 1.0},
            "crystal_structure": "fcc",
        },
        "dft": {"pseudopotential_dir": str(pseudo_dir), "ecutwfc": 30.0, "kspacing": 0.15},
        "runtime": {"work_dir": str(tmp_path / "work")},
    }

    config_file = tmp_path / "valid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["check-config", str(config_file)])
    assert result.exit_code == 0
    assert "Validation Successful" in result.stdout


def test_cli_check_config_failure(tmp_path):
    config_data = {
        "target_system": {
            # Missing elements
            "composition": {"Al": 1.0}
        }
    }
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["check-config", str(config_file)])
    assert result.exit_code == 1
    assert "Validation Error" in result.stdout


def test_cli_db_init(tmp_path):
    # Create dummy UPF dir
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    config_data = {
        "target_system": {"elements": ["Al"], "composition": {"Al": 1.0}},
        "dft": {"pseudopotential_dir": str(pseudo_dir), "ecutwfc": 30.0, "kspacing": 0.15},
        "runtime": {"database_path": str(tmp_path / "test.db")},
    }

    # We need to run db init.
    # Usually it reads from input.yaml or passed as arg?
    # Spec says "mlip-auto db init". It implies it reads default input.yaml or user provides it.
    # UAT Scenario 1.4: "Given a valid configuration file specifying mlip.db".
    # Does "mlip-auto db init" take a file arg?
    # Spec doesn't say. Let's assume it defaults to input.yaml or we can pass it.
    # CLI help in Spec: `mlip-auto check-config <file>`.
    # `mlip-auto db init` probably reads `input.yaml` by default.

    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open("input.yaml", "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        # The config specified an absolute path, so we check that path
        assert (tmp_path / "test.db").exists()
