from pathlib import Path

import pytest
from ase.io import read

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig


def test_orchestrator_mock_run(tmp_path: Path) -> None:
    """Test full execution of the mock loop."""
    work_dir = tmp_path / "work"

    config_data = {
        "orchestrator": {
            "work_dir": work_dir,
            "n_iterations": 2,
        },
        "generator": {
            "type": "mock",
            "n_candidates": 5,
        },
        "oracle": {
            "type": "mock",
        },
        "trainer": {
            "type": "mock",
        },
        "dynamics": {
            "type": "mock",
        },
        "validator": {
            "type": "mock",
        },
    }
    config = GlobalConfig.model_validate(config_data)

    # Use the work_dir from config, which is work_dir variable
    orch = Orchestrator(config, work_dir)
    orch.run_loop()

    # Verify state
    state = orch.state_manager.load()
    assert state.current_iteration == 2

    # Verify artifacts
    assert (work_dir / "iter_001" / "train.xyz").exists()
    assert (work_dir / "iter_002" / "train.xyz").exists()

    # Verify provenance (Cycle 1: Global, Cycle 2: Local)
    # Note: ASE reads tags into .info dictionary
    iter1_atoms = read(work_dir / "iter_001" / "train.xyz", index=":")
    assert isinstance(iter1_atoms, list)
    # We expect at least one atom
    assert len(iter1_atoms) > 0
    # Check source tag
    assert iter1_atoms[0].info["source"] == "mock_generator_global"

    iter2_atoms = read(work_dir / "iter_002" / "train.xyz", index=":")
    assert isinstance(iter2_atoms, list)
    assert len(iter2_atoms) > 0
    assert iter2_atoms[0].info["source"] == "mock_generator_local"


def test_orchestrator_init_error(tmp_path: Path) -> None:
    """Test Orchestrator raises NotImplementedError for non-mock components."""
    config_data = {
        "orchestrator": {"work_dir": tmp_path, "n_iterations": 1},
        "generator": {"type": "random", "elements": ["H"], "n_candidates": 10}, # Not implemented yet
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    config = GlobalConfig.model_validate(config_data)

    with pytest.raises(NotImplementedError):
        Orchestrator(config, tmp_path)
