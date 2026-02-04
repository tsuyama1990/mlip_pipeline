from pathlib import Path

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    GlobalConfig,
    TrainingConfig,
)

# We assume Orchestrator will be here
from mlip_autopipec.orchestration.orchestrator import Orchestrator


def test_orchestrator_run_mock() -> None:
    config = GlobalConfig(
        project_name="test_skeleton",
        execution_mode="mock",
        cycles=2,
        dft=DFTConfig(calculator="lj"),
        training=TrainingConfig(potential_type="ace"),
        exploration=ExplorationConfig(strategy="random", num_candidates=5),
    )

    orchestrator = Orchestrator(config)
    orchestrator.run()

    # Check artifacts
    assert Path("potential_001.yace").exists()
    assert Path("potential_002.yace").exists()

    # Cleanup
    if Path("potential_001.yace").exists():
        Path("potential_001.yace").unlink()
    if Path("potential_002.yace").exists():
        Path("potential_002.yace").unlink()
