from pathlib import Path

from src.config.config_model import GlobalConfig
from src.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from src.orchestration.orchestrator import Orchestrator


def test_skeleton_loop(tmp_path: Path) -> None:
    config = GlobalConfig(work_dir=tmp_path, max_cycles=2, random_seed=42)

    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()

    orchestrator = Orchestrator(
        config=config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
    )

    orchestrator.run()

    # Basic assertion that it finished (files created?)
    assert (tmp_path / "cycle_01").exists()
    assert (tmp_path / "cycle_02").exists()
