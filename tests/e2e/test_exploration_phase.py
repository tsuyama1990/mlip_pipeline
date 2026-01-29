"""E2E tests for Exploration Phase."""

from mlip_autopipec.domain_models.config import Config, ExplorationConfig
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase


def test_exploration_phase_execution() -> None:
    """Test that exploration phase generates candidates and updates state."""
    # Setup
    config = Config(
        project_name="test",
        structure_gen=ExplorationConfig(
            strategy="template",
            composition="Si",
            num_candidates=3
        )
    )
    state = WorkflowState(current_phase=WorkflowPhase.EXPLORATION)

    phase = ExplorationPhase()

    # Execute
    phase.execute(state, config)

    # Verify
    assert len(state.candidates) == 3
    assert state.candidates[0].source == "cold_start"

    # Verify candidate properties
    cand = state.candidates[0]
    assert "Si" in cand.formatted_formula
    assert cand.status == "PENDING"
