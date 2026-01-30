"""UAT Verification Script for Cycle 02."""

import logging
import stat
import sys
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path.cwd() / "src"))

from mlip_autopipec.domain_models.config import Config, LammpsConfig, PotentialConfig, ExplorationConfig, MDParams
from mlip_autopipec.orchestration.workflow import run_one_shot
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UAT-02")

def uat_02_02_missing_executable() -> bool:
    logger.info("Running UAT-02-02: Missing Executable Handling")

    config = Config(
        project_name="UAT_Missing",
        potential=PotentialConfig(elements=["Si"], cutoff=2.0),
        structure_gen=ExplorationConfig(composition="Si"),
        lammps=LammpsConfig(command="/path/to/nonexistent/lmp")
    )

    try:
        result = run_one_shot(config)
        # Should fail gracefully
        if result.status == JobStatus.FAILED:
            if "not found" in result.log_content or "No such file" in result.log_content:
                logger.info("UAT-02-02 Passed (Correctly detected missing executable)")
                return True
            else:
                 logger.error(f"Failed: Job failed but log doesn't mention missing exec. Log: {result.log_content}")
                 return False
        else:
             logger.error(f"Failed: Job status is {result.status}, expected FAILED")
             return False
    except Exception as e:
        logger.error(f"Failed: Raised unhandled exception: {e}")
        return False

def uat_02_01_one_shot_pipeline(tmp_path: Path) -> bool:
    logger.info("Running UAT-02-01: One-Shot MD Run (Mocked LAMMPS)")

    # Create a fake LAMMPS executable
    fake_lmp = tmp_path / "fake_lmp.sh"
    dump_file = "dump.lammpstrj"

    fake_script = f"""#!/bin/bash
    echo "LAMMPS (Mock) is running"
    # Create a dummy dump file
    cat > {dump_file} << EOF
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 5.43
0 5.43
0 5.43
ITEM: ATOMS id type x y z
1 1 0.1 0.1 0.1
2 1 1.5 1.5 1.5
EOF
    """
    fake_lmp.write_text(fake_script)
    fake_lmp.chmod(fake_lmp.stat().st_mode | stat.S_IEXEC)

    config = Config(
        project_name="UAT_Mocked",
        potential=PotentialConfig(elements=["Si"], cutoff=2.0),
        structure_gen=ExplorationConfig(
            composition="Si",
            md_params=MDParams(temperature=300, n_steps=100)
        ),
        lammps=LammpsConfig(command=str(fake_lmp.absolute()), timeout=5)
    )

    try:
        # We need to run inside tmp_path so the fake script writes dump there
        # But LammpsRunner creates a subfolder.
        # The fake script writes to CWD. LammpsRunner runs in a temp dir.
        # So the fake script will write to that temp dir.
        # But the fake script is invoked with absolute path.

        result = run_one_shot(config)

        if result.status == JobStatus.COMPLETED and isinstance(result, LammpsResult):
            if len(result.final_structure.positions) == 2:
                logger.info("UAT-02-01 Passed")
                return True
            else:
                logger.error("Failed: Final structure has wrong number of atoms")
                return False
        else:
            logger.error(f"Failed: Job status {result.status}. Log: {result.log_content}")
            return False

    except Exception as e:
        logger.error(f"Failed: Exception {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from pathlib import Path
    import tempfile

    success = True

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        success &= uat_02_02_missing_executable()
        success &= uat_02_01_one_shot_pipeline(tmp_path)

    if success:
        logger.info("All UAT tests passed.")
        sys.exit(0)
    else:
        logger.error("Some UAT tests failed.")
        sys.exit(1)
