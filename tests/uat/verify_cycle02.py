import logging
import shutil
import sys
from pathlib import Path

from ase import Atoms

# Add src to path
sys.path.append("src")

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_random_generation(tmp_path: Path) -> None:
    logger.info("--- Verifying Random Generation ---")
    tmp_path.mkdir(parents=True, exist_ok=True)
    seed_path = tmp_path / "seed_random.xyz"
    Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True).write(seed_path)  # type: ignore[no-untyped-call]

    config_dict = {
        "orchestrator": {"work_dir": tmp_path, "max_cycles": 1, "execution_mode": "mock"},
        "generator": {
            "type": "random",
            "seed_structure_path": str(seed_path),
            "mock_count": 5,
            "policy": {
                "strategy": "random",
                "strain_range": 0.05
            }
        },
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"}
    }
    # Validate config
    config = GlobalConfig.model_validate(config_dict)

    orch = Orchestrator(config)
    # Access generator directly or simulate exploration
    candidates = list(orch.generator.explore({"count": 5}))

    assert len(candidates) == 5
    assert candidates[0].provenance == "random"
    logger.info("✓ Random Generation Produced 5 candidates with correct provenance.")

def verify_adaptive_schedule(tmp_path: Path) -> None:
    logger.info("--- Verifying Adaptive Schedule ---")
    tmp_path.mkdir(parents=True, exist_ok=True)
    seed_path = tmp_path / "seed_adaptive.xyz"
    Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True).write(seed_path)  # type: ignore[no-untyped-call]

    policy = {
        "strategy": "adaptive",
        "temperature_schedule": [100.0, 500.0, 1000.0]
    }

    config_dict = {
        "orchestrator": {"work_dir": tmp_path, "max_cycles": 3, "execution_mode": "mock"},
        "generator": {
            "type": "adaptive",
            "seed_structure_path": str(seed_path),
            "policy": policy,
            "mock_count": 1
        },
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"}
    }
    config = GlobalConfig.model_validate(config_dict)

    orch = Orchestrator(config)

    # Cycle 0
    c0 = list(orch.generator.explore({"cycle": 0, "count": 1}))
    logger.info(f"Cycle 0 Provenance: {c0[0].provenance}")
    assert "md_100.0K" in c0[0].provenance

    # Cycle 1
    c1 = list(orch.generator.explore({"cycle": 1, "count": 1}))
    logger.info(f"Cycle 1 Provenance: {c1[0].provenance}")
    assert "md_500.0K" in c1[0].provenance

    # Cycle 2
    c2 = list(orch.generator.explore({"cycle": 2, "count": 1}))
    logger.info(f"Cycle 2 Provenance: {c2[0].provenance}")
    assert "md_1000.0K" in c2[0].provenance

    logger.info("✓ Adaptive Schedule verified.")

if __name__ == "__main__":
    base_tmp = Path("tests/uat/tmp_cycle02")
    if base_tmp.exists():
        shutil.rmtree(base_tmp)
    base_tmp.mkdir(parents=True, exist_ok=True)

    try:
        verify_random_generation(base_tmp / "random")
        verify_adaptive_schedule(base_tmp / "adaptive")
        logger.info("All Cycle 02 Verification Scenarios Passed!")
    except Exception:
        logger.exception("FAILED")
        sys.exit(1)
