import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"Running: {cmd}")  # noqa: T201
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)  # noqa: S602
    if check and res.returncode != 0:
        print(f"Error: {res.stderr}")  # noqa: T201
        sys.exit(1)
    return res


def test_cycle01() -> None:
    print("Starting UAT: Cycle 01")  # noqa: T201

    # Clean up previous run
    if Path("config.yaml").exists():
        Path("config.yaml").unlink()
    if Path("uat_run").exists():
        shutil.rmtree("uat_run")

    # 1. Init
    res = run_command("uv run mlip-runner init")
    assert "Created default configuration" in res.stdout
    assert Path("config.yaml").exists()

    # 2. Modify config
    with Path("config.yaml").open("r") as f:
        config = yaml.safe_load(f)

    work_dir = Path("uat_run")
    config["orchestrator"]["work_dir"] = str(work_dir)
    config["orchestrator"]["max_iterations"] = 3

    with Path("config.yaml").open("w") as f:
        yaml.dump(config, f)

    # 3. Run
    res = run_command("uv run mlip-runner run config.yaml")
    assert "Starting new run" in res.stdout
    assert work_dir.exists()

    # 4. Check state
    state_file = work_dir / "workflow_state.json"
    assert state_file.exists()

    # 5. Resume (simulate by running again)
    res = run_command("uv run mlip-runner run config.yaml")
    assert "Resuming from iteration" in res.stdout or "Starting new run" not in res.stdout

    print("UAT Cycle 01 Passed!")  # noqa: T201

    # Cleanup
    if Path("config.yaml").exists():
        Path("config.yaml").unlink()
    if Path("uat_run").exists():
        shutil.rmtree("uat_run")


if __name__ == "__main__":
    test_cycle01()
