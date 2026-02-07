import subprocess
from pathlib import Path


def test_scenario_01(tmp_path: Path) -> None:
    """
    SCENARIO 01: System Initialisation & Configuration
    Verify that the system can correctly parse a configuration file and initialise the necessary components.
    """
    workdir = tmp_path / "work"
    config_content = f"""
    workdir: "{workdir}"
    oracle:
      type: "mock"
      noise_level: 0.1
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    # Run command
    # Note: We use "uv run mlip-pipeline" to simulate end-user execution within the project environment.
    # The command structure is "mlip-pipeline run <config>"
    result = subprocess.run(
        ["uv", "run", "mlip-pipeline", "run", str(config_path)],
        capture_output=True,
        text=True,
    )

    # Check exit code
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check output logs
    assert "Configuration loaded successfully" in result.stdout
    assert "Initialised MockOrchestrator" in result.stdout
    assert "Pipeline finished" in result.stdout

    # Verify log file creation
    assert (workdir / "mlip.log").exists()
