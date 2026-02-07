from pathlib import Path

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GlobalConfig


def test_orchestrator_mock(tmp_path: Path) -> None:
    config = GlobalConfig(
        workdir=tmp_path / "test_run",
        max_cycles=2,
        generator={"type": "mock", "count": 2},
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"}
    )

    orch = Orchestrator(config)
    orch.run()

    assert (tmp_path / "test_run" / "potential_cycle_0.yace").exists()
    assert (tmp_path / "test_run" / "potential_cycle_1.yace").exists()
    assert len(orch.dataset) > 0

def test_orchestrator_no_initial_structures(tmp_path: Path) -> None:
    config = GlobalConfig(
        workdir=tmp_path / "test_run_empty",
        max_cycles=2,
        generator={"type": "mock", "count": 0},
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"}
    )

    orch = Orchestrator(config)
    orch.run()

    # Dataset should be empty
    assert len(orch.dataset) == 0
    # No potential should be created
    assert not (tmp_path / "test_run_empty" / "potential_cycle_0.yace").exists()
