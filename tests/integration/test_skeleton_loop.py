from pathlib import Path

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging


def test_skeleton_loop_execution(tmp_path: Path) -> None:
    # Setup logging to file
    log_file = tmp_path / "skeleton.log"
    setup_logging(level="DEBUG", log_file=str(log_file))

    config_data = {
        "project_name": "SkeletonProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 30.0,
            "kpoints": [1, 1, 1]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 4.0
        },
        "exploration": {
            "strategy": "random",
            "max_temperature": 300.0,
            "steps": 10
        }
    }
    config = SimulationConfig(**config_data) # type: ignore[arg-type]

    orchestrator = Orchestrator(config)

    # Run the cycle
    orchestrator.run()

    # Verify log content
    content = log_file.read_text()
    assert "Starting orchestration cycle..." in content
    assert "Phase 1: Exploration" in content
    assert "Phase 2: Oracle Labeling" in content
    assert "Phase 3: Training" in content
    assert "Phase 4: Validation" in content
    assert "Cycle completed successfully" in content
