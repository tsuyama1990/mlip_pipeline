import shutil
import subprocess
import sys
from pathlib import Path
import yaml

def verify_cycle01():
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
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    print(f"Running Cycle 01 UAT with config: {config_path}")

    # Run the command
    result = subprocess.run(
        [sys.executable, "-m", "mlip_autopipec.main", str(config_path)],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        print("❌ UAT Failed: Process exited with non-zero code.")
        sys.exit(1)

    # Verify outputs
    run_dir = uat_dir / "run"
    pot_dir = run_dir / "potentials"

    if not pot_dir.exists():
        print("❌ UAT Failed: Potentials directory not created.")
        sys.exit(1)

    potentials = list(pot_dir.glob("*.yace"))
    print(f"Found {len(potentials)} potentials: {[p.name for p in potentials]}")

    if len(potentials) < 1: # Expect at least 1 if mock trainer runs
        print("❌ UAT Failed: No potentials generated.")
        sys.exit(1)

    # Check log output for specific messages
    if "Orchestrator started" not in result.stderr and "Orchestrator started" not in result.stdout:
        # Logging goes to stdout/stderr depending on config. My logging setup sends to stdout.
        # Check stdout.
        if "Orchestrator started" not in result.stdout:
             print("❌ UAT Failed: 'Orchestrator started' log missing.")
             sys.exit(1)

    print("✅ Cycle 01 UAT Passed!")

if __name__ == "__main__":
    verify_cycle01()
