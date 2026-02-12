from pathlib import Path

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.datastructures import WorkflowState


def test_state_manager_init(tmp_path: Path) -> None:
    sm = StateManager(tmp_path)
    assert isinstance(sm.state, WorkflowState)
    assert sm.state.current_cycle == 0


def test_state_manager_save_load(tmp_path: Path) -> None:
    sm = StateManager(tmp_path)
    sm.update_cycle(5)

    # Reload
    sm2 = StateManager(tmp_path)
    assert sm2.state.current_cycle == 5


def test_state_manager_cleanup(tmp_path: Path) -> None:
    sm = StateManager(tmp_path)
    sm.save()

    # tmp file might exist during save but is renamed.
    # To test cleanup, we manually create a tmp file.
    (sm.state_file.with_suffix(".tmp")).touch()

    sm.cleanup()
    assert not (sm.state_file.with_suffix(".tmp")).exists()
