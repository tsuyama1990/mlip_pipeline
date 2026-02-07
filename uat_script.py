import os
import shutil
import subprocess
from pathlib import Path

import yaml


def run_uat():
    print("Running UAT Scenario...")
    workdir = Path("uat_run")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir()

    config_data = {
        "workdir": str(workdir / "run"),
        "max_cycles": 2,
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"}
    }

    config_file = workdir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    # Using uv run to execute main.py
    cmd = ["uv", "run", "python", "src/mlip_autopipec/main.py", "run", str(config_file)]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        print("UAT Failed: CLI execution failed")
        return False

    # Verify outputs
    pot0 = workdir / "run" / "potential_cycle_0.yace"
    pot1 = workdir / "run" / "potential_cycle_1.yace"

    if not pot0.exists():
        print(f"UAT Failed: {pot0} missing")
        return False
    if not pot1.exists():
        print(f"UAT Failed: {pot1} missing")
        return False

    print("UAT Passed!")
    return True

if __name__ == "__main__":
    success = run_uat()
    if not success:
        exit(1)
