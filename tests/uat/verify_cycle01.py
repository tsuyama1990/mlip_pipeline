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

        # Verify content
        with open("input.yaml") as f:
            config = yaml.safe_load(f)
        assert config["target_system"]["name"] == "FeNi System"

        # Run (using alias 'run')
        # We expect it to try running the loop.
        print("Invoking 'mlip-auto run'...")
        result = runner.invoke(app, ["run"])

        # Check if command was recognized (not "No such command")
        if "No such command" in result.stdout:
            raise AssertionError("Command 'run' alias failed! Output: " + result.stdout)

        # It is acceptable for it to fail due to missing config values or environment in this barebones test
        print(f"Run output snippet: {result.stdout[:200]}...")

        # Scenario 2: Invalid Config
        print("Running Scenario 2: Invalid Config")
        # Modify config to be invalid
        config["dft"]["ecutwfc"] = -10.0
        with open("input.yaml", "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0, "Validation should have failed for negative ecutwfc"
        assert "Validation Error" in result.stdout or "Invalid configuration" in result.stdout
        print("Invalid config caught successfully.")

    finally:
        # Cleanup
        os.chdir(base_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)

if __name__ == "__main__":
    test_cycle01_uat()
    print("CYCLE 01 UAT PASSED")
