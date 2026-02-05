import subprocess
from pathlib import Path

import yaml


def test_uat_loop_execution(tmp_path: Path) -> None:
    """
    User Acceptance Test:
    Verify that the user can run the full pipeline via the CLI command.
    """
    config_data = {
        "work_dir": str(tmp_path),
        "logging_level": "INFO",
        "max_cycles": 1,
        "exploration": {"max_structures": 2}
    }
    config_file = tmp_path / "uat_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Run via CLI using uv
    # explicit check=False to satisfy linter
    cmd = ["uv", "run", "mlip-pipeline", "run", str(config_file)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False) # noqa: S603

    if result.returncode != 0:
        # Logging instead of print would be better, but for test failure debug:
        pass

    assert result.returncode == 0
    assert "Pipeline completed successfully" in result.stdout
    assert (tmp_path / "potential.yace").exists()
    assert (tmp_path / "mlip.log").exists()
