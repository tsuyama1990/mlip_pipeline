"""UAT Verification Script for Cycle 02."""

import logging
import sys
from pathlib import Path

# Ensure src is in path for standalone execution
sys.path.append(str(Path.cwd() / "src"))

from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.config import Config, PotentialConfig, LammpsConfig

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UAT-02")

runner = CliRunner()

def setup_config(path: Path, bad_cmd: bool = False, timeout: float = 3600.0) -> None:
    config = Config(
        project_name="UAT_Cycle02",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig(
            command="echo 'LAMMPS MOCK' " if not bad_cmd else "non_existent_command",
            cores=1,
            timeout=timeout
        )
    )
    from mlip_autopipec.infrastructure import io
    io.dump_yaml(config.model_dump(mode='json'), path)


def create_fake_lammps_script(path: Path):
    """Create a fake lammps executable that produces valid output."""
    script = """#!/bin/bash
# Mock LAMMPS
# Expected args: -in in.lammps
# We should parse args if needed, but for now just write output.

echo "LAMMPS Version 2023"
# Simulate run time
sleep 0.1

# Write dump.lammpstrj
cat <<EOF > dump.lammpstrj
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 5.43
0.0 5.43
0.0 5.43
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.35 1.35 1.35
EOF
"""
    path.write_text(script)
    path.chmod(0o755)


def uat_02_01_one_shot_success(tmp_path: Path) -> bool:
    logger.info("Running UAT-02-01: One-Shot Success")

    config_path = tmp_path / "config.yaml"
    fake_lammps = tmp_path / "fake_lammps.sh"
    create_fake_lammps_script(fake_lammps)

    # Setup config pointing to fake lammps
    config = Config(
        project_name="UAT_Cycle02",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig(
            command=str(fake_lammps.absolute()),
            cores=1
        )
    )
    from mlip_autopipec.infrastructure import io
    # Use mode='json' to serialize Paths to strings
    io.dump_yaml(config.model_dump(mode='json'), config_path)

    result = runner.invoke(app, ["run-cycle-02", "--config", str(config_path)])

    if result.exit_code != 0:
        logger.error(f"CLI failed with code {result.exit_code}")
        logger.error(result.stdout)
        return False

    if "Simulation Completed: Status COMPLETED" not in result.stdout:
        logger.error("Output did not contain success message")
        logger.error(result.stdout)
        return False

    logger.info("UAT-02-01 Passed")
    return True


def uat_02_02_missing_executable(tmp_path: Path) -> bool:
    logger.info("Running UAT-02-02: Missing Executable")

    config_path = tmp_path / "config_bad.yaml"

    config = Config(
        project_name="UAT_Cycle02",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig(
            command="/non/existent/path/lmp",
            cores=1
        )
    )
    from mlip_autopipec.infrastructure import io
    io.dump_yaml(config.model_dump(mode='json'), config_path)

    result = runner.invoke(app, ["run-cycle-02", "--config", str(config_path)])

    # Expect failure
    if result.exit_code == 0:
        logger.error("CLI succeeded unexpectedly")
        return False

    if "Execution failed" not in result.stdout:
        logger.error("Output did not contain expected error message")
        logger.error(result.stdout)
        return False

    logger.info("UAT-02-02 Passed")
    return True


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        success = True
        success &= uat_02_01_one_shot_success(tmp_path)
        success &= uat_02_02_missing_executable(tmp_path)

    if success:
        logger.info("All UAT tests passed.")
        sys.exit(0)
    else:
        logger.error("Some UAT tests failed.")
        sys.exit(1)
