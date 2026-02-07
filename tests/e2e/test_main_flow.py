import subprocess
import sys
from pathlib import Path


def test_dry_run_main_flow(tmp_path: Path) -> None:
    # Setup test directory
    work_dir = tmp_path / "work_dir"
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

    # Run the main script using sys.executable to ensure we use the same python
    # Safe to suppress: This test runs the tool itself using the current Python interpreter
    # on a config file created within the test harness
    result = subprocess.run(
        [sys.executable, "-m", "mlip_autopipec.main", str(config_path)],
        capture_output=True,
        text=True,
        check=False
    )

    assert result.returncode == 0

    # Check if potentials directory exists
    pot_dir = work_dir / "potentials"
    assert pot_dir.exists()
    assert any(pot_dir.glob("*.yace"))
