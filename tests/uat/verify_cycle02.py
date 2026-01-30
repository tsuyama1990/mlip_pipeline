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
    # Test hybrid potential config
    config = Config(
        project_name="UAT_Cycle02",
        potential=PotentialConfig(
            elements=["Si"],
            cutoff=5.0,
            lattice_constant=5.43,
            crystal_structure="diamond",
            pair_style="hybrid/overlay ace lj/cut 2.5",
            pair_coeff=["* * ace potential.yace Si", "* * lj/cut 1.0 1.0"]
        ),
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

    # Verify input script contains new potential info
    # We need to find the job directory. It's printed in output "Result in: ..."
    # Or we can look in _work_md/
    import re
    match = re.search(r"Result in: (.*)", result.stdout)
    if match:
        work_dir = Path(match.group(1).strip())
        in_lammps = work_dir / "in.lammps"
        if in_lammps.exists():
            content = in_lammps.read_text()
            if "hybrid/overlay ace" not in content:
                logger.error("Input script does not contain hybrid potential")
                return False
            if "potential.yace" not in content:
                logger.error("Input script does not contain pair coeff")
                return False
        else:
            logger.warning("Could not find in.lammps to verify content")

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

    # We need to set the CWD to tmp_dir for UAT because LammpsRunner uses _work_md relative to CWD
    # But LammpsRunner uses base_work_dir which defaults to _work_md
    # In tests, we might want to override base_work_dir, but via CLI we can't yet (not exposed).
    # So _work_md will be created in CWD.
    # To keep UAT clean, we can chdir to tmp_dir?
    import os

    original_cwd = os.getcwd()

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        os.chdir(tmp_path)
        try:
            success = True
            success &= uat_02_01_one_shot_success(tmp_path)
            success &= uat_02_02_missing_executable(tmp_path)
        finally:
            os.chdir(original_cwd)

    if success:
        logger.info("All UAT tests passed.")
        sys.exit(0)
    else:
        logger.error("Some UAT tests failed.")
        sys.exit(1)
