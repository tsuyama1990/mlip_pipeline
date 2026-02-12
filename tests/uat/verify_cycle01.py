import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def run_command(cmd: list[str], cwd: Path | None = None, expect_fail: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"Running: {' '.join(cmd)}")  # noqa: T201
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=False
    )
    if not expect_fail and result.returncode != 0:
        print(f"STDOUT: {result.stdout}")  # noqa: T201
        print(f"STDERR: {result.stderr}")  # noqa: T201
        msg = f"Command failed: {cmd}"
        raise RuntimeError(msg)
    if expect_fail and result.returncode == 0:
        print(f"STDOUT: {result.stdout}")  # noqa: T201
        msg = f"Command succeeded unexpectedly: {cmd}"
        raise RuntimeError(msg)
    return result

def verify_cycle01() -> None:
    print("=== Starting Cycle 01 UAT ===")  # noqa: T201

    work_dir = Path("uat_work_dir")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    python_exe = sys.executable

    # Scenario 1: System Initialization
    print("\n--- Scenario 1: System Initialization ---")  # noqa: T201
    run_command([python_exe, "-m", "mlip_autopipec.main", "init"], cwd=work_dir)

    config_path = work_dir / "config.yaml"
    assert config_path.exists(), "config.yaml not created"

    with config_path.open() as f:
        config = yaml.safe_load(f)
    assert "orchestrator" in config
    print("✓ Init successful")  # noqa: T201

    # Scenario 2: Configuration Validation
    print("\n--- Scenario 2: Configuration Validation ---")  # noqa: T201
    bad_config_path = work_dir / "bad_config.yaml"
    bad_config_data = copy.deepcopy(config)
    bad_config_data["orchestrator"]["max_iterations"] = "invalid"

    with bad_config_path.open("w") as f:
        yaml.dump(bad_config_data, f)

    res = run_command([python_exe, "-m", "mlip_autopipec.main", "run", "bad_config.yaml"], cwd=work_dir, expect_fail=True)
    # Check stdout or stderr depending on where typer prints
    output = res.stdout + res.stderr
    assert "validation error" in output.lower()
    print("✓ Invalid config rejected")  # noqa: T201

    # Scenario 3: State Persistence
    print("\n--- Scenario 3: State Persistence ---")  # noqa: T201
    # Run correctly
    # Update config work_dir to absolute path inside uat_work_dir/run
    run_dir = work_dir / "run"
    config["orchestrator"]["work_dir"] = str(run_dir.absolute())

    # Save valid config
    valid_config_path = work_dir / "valid_config.yaml"
    with valid_config_path.open("w") as f:
        yaml.dump(config, f)

    run_command([python_exe, "-m", "mlip_autopipec.main", "run", "valid_config.yaml"], cwd=work_dir)

    state_file = run_dir / "workflow_state.json"
    assert state_file.exists(), "State file not created"

    with state_file.open() as f:
        state = json.load(f)
    assert state["iteration"] == 0

    # Run again (Resume)
    res = run_command([python_exe, "-m", "mlip_autopipec.main", "run", "valid_config.yaml"], cwd=work_dir)
    assert "Resuming from iteration 0" in res.stdout
    print("✓ State persistence verified")  # noqa: T201

    print("\n=== Cycle 01 UAT Passed ===")  # noqa: T201

    # Cleanup
    if work_dir.exists():
        shutil.rmtree(work_dir)

if __name__ == "__main__":
    verify_cycle01()
