"""UAT Verification Script for Cycle 02."""

import logging
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure src is in path
sys.path.append(str(Path.cwd() / "src"))

from mlip_autopipec.domain_models.config import Config, PotentialConfig, LammpsConfig
from mlip_autopipec.orchestration import workflow
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UAT-02")

def uat_02_01_one_shot_execution() -> bool:
    logger.info("Running UAT-02-01: One-Shot MD Run")

    # 1. Configure
    config = Config(
        project_name="UAT_02",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig(command="lmp_mock", timeout=10)
    )

    # 2. Mock LAMMPS execution since we don't have binary
    with patch("mlip_autopipec.physics.dynamics.lammps.subprocess.run") as mock_run, \
         patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output") as mock_parse:

        mock_run.return_value = MagicMock(returncode=0, stdout="Simulation done", stderr="")

        # Mock parsing to return success
        # Create a real Structure object for validation
        real_structure = Structure(
            symbols=["Si"] * 8,
            positions=np.zeros((8, 3)),
            cell=np.eye(3) * 5.43,
            pbc=(True, True, True)
        )

        mock_parse.return_value = LammpsResult(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            work_dir=Path("_work_md/test_job"),
            duration_seconds=1.0,
            log_content="Simulation done",
            final_structure=real_structure
        )

        try:
            workflow.run_one_shot(config)
            logger.info("UAT-02-01 Passed")
            return True
        except Exception as e:
            logger.error(f"UAT-02-01 Failed: {e}", exc_info=True)
            return False

def uat_02_02_missing_executable() -> bool:
    logger.info("Running UAT-02-02: Missing Executable Handling")

    # This scenario is tricky because our code assumes subprocess just runs the command string.
    # If the executable is missing, shell=True in subprocess.run might return 127.

    config = Config(
        project_name="UAT_02",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig(command="non_existent_executable", timeout=10)
    )

    # We want to verify that it fails gracefully
    # But since we are likely not running this in a real shell with shell=True capable of finding "non_existent_executable",
    # we can rely on our previous mock strategy or try to run it for real if shell is available.
    # However, for robustness in this environment, let's mock subprocess returning 127

    with patch("mlip_autopipec.physics.dynamics.lammps.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=127, stdout="", stderr="command not found")

        try:
            workflow.run_one_shot(config)
            # It should raise RuntimeError
            logger.error("UAT-02-02 Failed: Should have raised RuntimeError")
            return False
        except RuntimeError as e:
            if "Job failed with status" in str(e):
                logger.info("UAT-02-02 Passed: Gracefully handled failure")
                return True
            else:
                logger.error(f"UAT-02-02 Failed: Unexpected error {e}")
                return False

if __name__ == "__main__":
    success = True
    success &= uat_02_01_one_shot_execution()
    success &= uat_02_02_missing_executable()

    if success:
        logger.info("All UAT tests passed.")
        sys.exit(0)
    else:
        logger.error("Some UAT tests failed.")
        sys.exit(1)
