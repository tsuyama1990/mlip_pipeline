from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_uat_01_01_hello_world() -> None:
    config = {
        "project_name": "uat_test",
        "execution_mode": "mock",
        "cycles": 2,
        "dft": {"calculator": "lj", "kpoints_density": 0.04, "encut": 500.0},
        "training": {"potential_type": "ace", "cutoff": 5.0, "max_degree": 1},
        "exploration": {"strategy": "random", "num_candidates": 5, "supercell_size": 2},
    }

    with Path("uat_config.yaml").open("w") as f:
        yaml.dump(config, f)

    result = runner.invoke(app, ["run", "uat_config.yaml"])

    # Clean up first to avoid leaving junk even if test fails
    if Path("uat_config.yaml").exists():
        Path("uat_config.yaml").unlink()
    if Path("potential_001.yace").exists():
        Path("potential_001.yace").unlink()
    if Path("potential_002.yace").exists():
        Path("potential_002.yace").unlink()

    assert result.exit_code == 0
    assert "Cycle 1" in result.stdout
    assert "Cycle 2" in result.stdout


def test_uat_01_02_config_validation() -> None:
    config = {
        "project_name": "uat_fail",
        "execution_mode": "mock",
        "cycles": 1,
        "dft": {"calculator": "lj", "kpoints_density": 0.04, "encut": 500.0},
        "training": {
            "potential_type": "ace",
            "cutoff": -5.0,  # Invalid
            "max_degree": 1,
        },
        "exploration": {"strategy": "random", "num_candidates": 5, "supercell_size": 2},
    }

    with Path("bad_config.yaml").open("w") as f:
        yaml.dump(config, f)

    result = runner.invoke(app, ["run", "bad_config.yaml"])

    if Path("bad_config.yaml").exists():
        Path("bad_config.yaml").unlink()

    assert result.exit_code != 0
    assert (
        "Input should be greater than 0" in str(result.stdout)
        or "validation error" in str(result.stdout).lower()
    )
