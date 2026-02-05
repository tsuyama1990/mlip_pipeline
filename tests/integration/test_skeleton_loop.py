from pathlib import Path

from mlip_autopipec.config.config_model import ExplorationConfig, GlobalConfig
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator


def test_skeleton_loop(tmp_path: Path) -> None:
    # Construct sub-config object to satisfy type checker
    exp_config = ExplorationConfig(max_structures=2)

    config = GlobalConfig(
        work_dir=tmp_path,
        max_cycles=2,
        exploration=exp_config
    )

    explorer = MockExplorer(config)
    oracle = MockOracle(config)
    trainer = MockTrainer(config)
    validator = MockValidator(config)

    orch = Orchestrator(config, explorer, oracle, trainer, validator)
    orch.run()

    # Verify outputs
    # 2 cycles * 2 structures = 4 structures
    assert len(orch.dataset) == 4

    # Check artifacts
    assert (tmp_path / "potential.yace").exists()

    # I'll rely on Orchestrator logic.
    assert orch.dataset._structures[0].source == "mock_explorer"
