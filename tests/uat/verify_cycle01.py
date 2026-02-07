import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Configure logger for UAT
logger = logging.getLogger("uat_cycle01")
logging.basicConfig(level=logging.INFO)

def verify_cycle01() -> None:
    uat_dir = Path("uat_cycle01")
    if uat_dir.exists():
        shutil.rmtree(uat_dir)
    uat_dir.mkdir()

    config_content = {
        "work_dir": str(uat_dir / "run"),
        "max_cycles": 3,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "explorer": {"type": "mock"}
    }

    config_path = uat_dir / "config.yaml"
    config_path.write_text(yaml.dump(config_content))

    logger.info(f"Running Cycle 01 UAT with config: {config_path}")

    # Run the command
    # Safe to suppress: This UAT test runs the CLI with a hardcoded config
    # generated within the test environment
    result = subprocess.run( # noqa: S603
        [sys.executable, "-m", "mlip_autopipec.main", str(config_path)],
        capture_output=True,
        text=True,
        check=False
    )

    logger.info(f"STDOUT: {result.stdout}")
    logger.info(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        logger.error("❌ UAT Failed: Process exited with non-zero code.")
        sys.exit(1)

    # Verify outputs
    run_dir = uat_dir / "run"
    pot_dir = run_dir / "potentials"

    if not pot_dir.exists():
        logger.error("❌ UAT Failed: Potentials directory not created.")
        sys.exit(1)

    potentials = list(pot_dir.glob("*.yace"))
    logger.info(f"Found {len(potentials)} potentials: {[p.name for p in potentials]}")

    if len(potentials) < 1: # Expect at least 1 if mock trainer runs
        logger.error("❌ UAT Failed: No potentials generated.")
        sys.exit(1)

    # Check log output for specific messages
    # Logging goes to stdout/stderr depending on config. My logging setup sends to stdout.
    # Check stdout.
    if "Orchestrator started" not in result.stdout:
         logger.error("❌ UAT Failed: 'Orchestrator started' log missing.")
         sys.exit(1)

    logger.info("✅ Cycle 01 UAT Passed!")

if __name__ == "__main__":
    verify_cycle01()
