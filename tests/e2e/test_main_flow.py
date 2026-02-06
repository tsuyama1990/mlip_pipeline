import subprocess
import os
from pathlib import Path

def test_dry_run_main_flow() -> None:
    # Setup test directory
    work_dir = Path("tests/e2e/work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    config_content = f"""
work_dir: "{work_dir}"
max_cycles: 2
oracle:
  type: "mock"
trainer:
  type: "mock"
explorer:
  type: "mock"
"""
    config_path = work_dir / "config.yaml"
    config_path.write_text(config_content)

    # Run the main script
    # We use python -m mlip_autopipec.main config.yaml
    result = subprocess.run(
        ["python", "-m", "mlip_autopipec.main", str(config_path)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0

    # Verify outputs
    # Mock trainer should produce a potential file.
    # Mock oracle should log something.

    # Check if potentials directory exists
    pot_dir = work_dir / "potentials"
    assert pot_dir.exists()
    assert any(pot_dir.glob("*.yace"))

    # Clean up (optional)
    # import shutil
    # shutil.rmtree(work_dir)
