import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_uat() -> None:
    root = Path(__file__).parents[2]
    src = root / "src"

    # Update PYTHONPATH to include src
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + ":" + env.get("PYTHONPATH", "")

    main_script = src / "mlip_autopipec" / "main.py"
    work_dir = root / "uat_work"
    config_file = work_dir / "config.yaml"

    # Cleanup
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    print("Running init...") # noqa: T201
    subprocess.run([sys.executable, str(main_script), "init", "--work-dir", str(work_dir), "--config-file", str(config_file)], check=True, env=env) # noqa: S603

    if not config_file.exists():
        print("FAIL: config file not created") # noqa: T201
        sys.exit(1)

    print("Running run-loop...") # noqa: T201
    subprocess.run([sys.executable, str(main_script), "run-loop", "--config-file", str(config_file)], check=True, env=env) # noqa: S603

    if not (work_dir / "workflow_state.json").exists():
        print("FAIL: workflow_state.json not created") # noqa: T201
        sys.exit(1)

    print("UAT Passed!") # noqa: T201

if __name__ == "__main__":
    run_uat()
