from pathlib import Path

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import ComponentConfig, GlobalConfig


def test_orchestrator_initialization() -> None:
    """Test initializing the Orchestrator."""
    config = GlobalConfig(
        workdir=Path("runs/test"),
        max_cycles=1,
        generator=ComponentConfig(type="mock"),
        oracle=ComponentConfig(type="mock"),
        trainer=ComponentConfig(type="mock"),
        dynamics=ComponentConfig(type="mock"),
        validator=ComponentConfig(type="mock"),
    )
    orchestrator = Orchestrator(config)
    assert orchestrator.config == config


def test_orchestrator_run_mock(tmp_path: Path) -> None:
    """Test running the Orchestrator with mock components."""
    # Use tmp_path for workdir
    workdir = tmp_path / "runs" / "test"
    # Ensure parent exists, but let Orchestrator create workdir or assume it exists
    workdir.mkdir(parents=True, exist_ok=True)

    config = GlobalConfig(
        workdir=workdir,
        max_cycles=1,
        generator=ComponentConfig(type="mock"),
        oracle=ComponentConfig(type="mock"),
        trainer=ComponentConfig(type="mock"),
        dynamics=ComponentConfig(type="mock"),
        validator=ComponentConfig(type="mock"),
    )
    orchestrator = Orchestrator(config)
    orchestrator.run()

    # Check if files created
    # MockTrainer creates "mock_potential.yace" inside workdir
    # Wait, MockTrainer logic uses workdir / "mock_potential.yace" if workdir provided.
    # The Orchestrator should pass workdir to Trainer.
    assert (workdir / "mock_potential.yace").exists()
