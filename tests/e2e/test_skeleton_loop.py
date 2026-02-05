from pathlib import Path

import yaml

try:
    from config import GlobalConfig
    from orchestration.orchestrator import Orchestrator
except ImportError:
    Orchestrator = None
    GlobalConfig = None

def test_skeleton_loop_execution(tmp_path):
    # Setup config
    config_data = {"work_dir": str(tmp_path), "max_cycles": 2, "random_seed": 42}
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    # Load Config
    config = GlobalConfig(**config_data)

    orchestrator = Orchestrator(config=config)
    orchestrator.run_loop()

    # Assertions
    assert len(orchestrator.dataset.structures) > 0
    assert orchestrator.potential_path == Path("mock_potential.pwo")
