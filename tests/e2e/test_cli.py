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
            "crystal_structure": "fcc"
        },
        "dft": {
            "pseudopotential_dir": str(pseudo_dir),
            "ecutwfc": 30.0,
            "kspacing": 0.15
        },
        "runtime": {
            "work_dir": str(tmp_path / "work")
        }
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

def test_cli_db_init_absolute_path(tmp_path):
    # Create dummy UPF dir
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    # Use a clearly absolute path
    abs_db_path = str(tmp_path.resolve() / "absolute_test.db")

    config_data = {
        "target_system": {"elements": ["Al"], "composition": {"Al": 1.0}},
        "dft": {"pseudopotential_dir": str(pseudo_dir), "ecutwfc": 30.0, "kspacing": 0.15},
        "runtime": {"database_path": abs_db_path}
    }

    # Run in a different CWD to ensure absolute path is respected
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()

    with runner.isolated_filesystem(temp_dir=cwd_dir):
        # Write config in cwd (or passed via path)
        with open("input.yaml", "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        # Check that db is NOT in cwd
        assert not Path("absolute_test.db").exists()
        # Check that db IS at absolute path
        assert Path(abs_db_path).exists()

def test_cli_db_init_relative_path(tmp_path):
    # Create dummy UPF dir
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    # Use relative path "local.db"
    config_data = {
        "target_system": {"elements": ["Al"], "composition": {"Al": 1.0}},
        "dft": {"pseudopotential_dir": str(pseudo_dir), "ecutwfc": 30.0, "kspacing": 0.15},
        "runtime": {"database_path": "local.db"}
    }

    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open("input.yaml", "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        # Check relative path
        assert Path("local.db").exists()

def test_cli_db_init_non_absolute_path_explicit(tmp_path):
    """Explicitly test a non-absolute path logic as requested by auditor."""
    # This mirrors relative_path test but ensures semantics
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    db_name = "subdir/my.db"
    config_data = {
        "target_system": {"elements": ["Al"], "composition": {"Al": 1.0}},
        "dft": {"pseudopotential_dir": str(pseudo_dir), "ecutwfc": 30.0, "kspacing": 0.15},
        "runtime": {"database_path": db_name}
    }

    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open("input.yaml", "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        assert (Path.cwd() / db_name).exists()
