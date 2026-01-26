import os
import shutil
from pathlib import Path
from typer.testing import CliRunner
from mlip_autopipec.app import app
import yaml

runner = CliRunner()

def test_cycle01_uat():
    # Setup workspace
    base_dir = Path.cwd()
    work_dir = base_dir / "uat_workspace_cycle01"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    os.chdir(work_dir)

    try:
        # Scenario 1: First-Time Setup
        print("Running Scenario 1: First-Time Setup")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, f"Init failed: {result.stdout}"
        assert Path("input.yaml").exists(), "input.yaml not created"

        # Verify content integrity
        with open("input.yaml") as f:
            config = yaml.safe_load(f)
        assert config["target_system"]["name"] == "FeNi System"
        assert config["workflow"]["max_generations"] == 5

        # Scenario 1b: Verify run loop behavior (dry run simulation)
        # We expect it to try running the loop.
        print("Invoking 'mlip-auto run'...")
        result = runner.invoke(app, ["run"])

        # Check if command was recognized (not "No such command")
        if "No such command" in result.stdout:
            raise AssertionError("Command 'run' alias failed! Output: " + result.stdout)

        # It fails because config points to paths that don't exist (e.g. upf), which is CORRECT behavior for validation
        assert "Validation Error" in result.stdout or "Workflow Failed" in result.stdout
        print("Run attempted and validated config successfully.")

        # Scenario 2: Invalid Config Security/Validation
        print("Running Scenario 2: Invalid Config")
        # Modify config to be invalid
        config["dft"]["ecutwfc"] = -10.0
        with open("input.yaml", "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0, "Validation should have failed for negative ecutwfc"
        assert "Validation Error" in result.stdout or "Invalid configuration" in result.stdout
        print("Invalid config caught successfully.")

        # Scenario 3: Large Config File (DoS Protection)
        print("Running Scenario 3: Large Config File")
        large_file = Path("large.yaml")
        # Write actual data to ensure st_size is correct (avoid sparse file issues)
        # 11 MB of spaces
        chunk = b" " * 1024 * 1024
        with open(large_file, "wb") as f:
            for _ in range(11):
                f.write(chunk)

        print(f"File size: {large_file.stat().st_size}")
        result = runner.invoke(app, ["validate", str(large_file)])
        print(f"Large File Result Output: {result.stdout}")
        assert result.exit_code != 0
        # We strictly want the size error now
        assert "too large" in result.stdout
        print("Large file rejection verified.")

    finally:
        # Cleanup
        os.chdir(base_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)

if __name__ == "__main__":
    test_cycle01_uat()
    print("CYCLE 01 UAT PASSED")
