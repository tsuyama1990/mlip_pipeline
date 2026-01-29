from pathlib import Path

import pytest

from mlip_autopipec.domain_models.config import Config, PotentialConfig
from mlip_autopipec.domain_models.workflow import WorkflowPhase
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def config() -> Config:
    return Config(
        project_name="test",
        potential=PotentialConfig(elements=["H"], cutoff=1.0)
    )

def test_workflow_manager_init(config: Config) -> None:
    wm = WorkflowManager(config)
    assert wm.state.current_phase == WorkflowPhase.INITIALIZATION
    assert wm.state.cycle_index == 0

def test_workflow_manager_run(config: Config, tmp_path: Path) -> None:
    wm = WorkflowManager(config)
    # Override state path to use temp dir
    wm.state_path = tmp_path / "state.json"

    wm.run()

    assert wm.state.is_halted is False
    assert (tmp_path / "state.json").exists()

    # Run again, should load state
    wm2 = WorkflowManager(config)
    wm2.state_path = tmp_path / "state.json"
    wm2.run()

    # Verify it loaded the state (cycle index should still be 0 as we didn't increment)
    assert wm2.state.cycle_index == 0
