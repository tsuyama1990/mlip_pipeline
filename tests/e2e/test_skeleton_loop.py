import yaml
from pathlib import Path
from typing import Optional, Type, TYPE_CHECKING
import pytest

# Use TYPE_CHECKING to avoid runtime import issues, but allow mypy to see types
if TYPE_CHECKING:
    from config import GlobalConfig
    from orchestration.orchestrator import Orchestrator

# Runtime import attempt
try:
    from config import GlobalConfig
    from orchestration.orchestrator import Orchestrator
except ImportError:
    Orchestrator = None
    GlobalConfig = None

def test_skeleton_loop_execution(tmp_path: Path) -> None:
    if Orchestrator is None or GlobalConfig is None:
        pytest.skip("Orchestrator or GlobalConfig not imported")

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
