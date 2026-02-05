import yaml
from pathlib import Path
from typing import Optional, Type, TYPE_CHECKING, Any
import pytest

if TYPE_CHECKING:
    from config import GlobalConfig
    from orchestration.orchestrator import Orchestrator

# Runtime import attempt
try:
    from config import GlobalConfig
    from orchestration.orchestrator import Orchestrator
    MODULES_IMPORTED = True
except ImportError:
    MODULES_IMPORTED = False

def test_skeleton_loop_execution(tmp_path: Path) -> None:
    if not MODULES_IMPORTED:
        pytest.skip("Orchestrator or GlobalConfig not imported")

    # Setup config
    config_data = {"work_dir": str(tmp_path), "max_cycles": 2, "random_seed": 42}
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    # Load Config
    # We cast config_data to Any to avoid mypy complaining about kwargs unpacking
    # strictly against the GlobalConfig signature which expects typed args.
    # In a real app, we'd validate the dict before unpacking or rely on Pydantic's runtime validation.
    config = GlobalConfig(**config_data) # type: ignore[arg-type]

    orchestrator = Orchestrator(config=config)
    orchestrator.run_loop()

    # Assertions
    assert len(orchestrator.dataset.structures) > 0
    assert orchestrator.potential_path == Path("mock_potential.pwo")
