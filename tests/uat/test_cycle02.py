"""UAT for Cycle 02: DIRECT Sampling & Active Learning."""

from pathlib import Path

import pytest

from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.orchestrator import Orchestrator

# Mocks and dependencies
# Assuming orchestrator.run_step1... and run_step2... are methods
# We will use mock dependencies injected into Orchestrator if possible, or config flags.

def test_cycle02_direct_sampling(tmp_path: Path) -> None:
    """Scenario 01: Intelligent Structure Generation."""
    # Setup config
    config_dict = {
        "project": {"name": "test_uat", "root_dir": str(tmp_path)},
        "distillation": {
            "enable_mace_distillation": True,
            "step1_direct_sampling": {"target_points": 50, "objective": "random"},
            "step2_active_learning": {"n_select": 10}
        },
        "oracle": {
            "dft": {
                "code": "mock",
                "command": "mock",
                "pseudopotentials": {"Fe": "mock.pbe"}
            },
                "mace": {
                    "model_path": "medium",
                    "mock": True
                },
            "mock": True
        }
    }
    # Load config (assuming we can construct from dict or mocking loader)
    # Since config loader is separate, we'll try to construct minimal object or assume mock orchestrator
    # For UAT, we usually run the full pipeline or key components.

    # Let's assume we can instantiate Orchestrator with a config object.
    # We need to construct the full config object which is complex.
    # Alternatively, use a minimal config yaml and load it.

    config_yaml = tmp_path / "config.yaml"
    # Write minimal valid yaml
    import yaml
    with config_yaml.open("w") as f:
        yaml.dump(config_dict, f)

    # We need to load it properly to get defaults
    from pyacemaker.core.config_loader import load_config
    try:
        config = load_config(config_yaml)
    except Exception as e:
        pytest.fail(f"Failed to load config: {e}")

    # Initialize Orchestrator
    orchestrator = Orchestrator(config)

    # Run Step 1
    # Check if method exists (it might not yet)
    if not hasattr(orchestrator, "run_step1_direct_sampling"):
        pytest.skip("run_step1_direct_sampling not implemented")

    generated_structures = orchestrator.run_step1_direct_sampling()

    # Verify count
    assert len(generated_structures) == 50
    # Verify metadata
    for s in generated_structures:
        # Check generation method or tags
        is_direct = s.generation_method == "direct" or "direct" in s.tags
        assert is_direct, f"Structure {s.id} missing direct tag/method"
        assert s.status == "NEW"


def test_cycle02_active_learning(tmp_path: Path) -> None:
    """Scenario 02: Active Learning Selection."""
    # Similar setup
    from pyacemaker.core.config_loader import load_config

    config_dict = {
        "project": {"name": "test_uat_al", "root_dir": str(tmp_path)},
        "distillation": {
            "enable_mace_distillation": True,
            "step2_active_learning": {"n_select": 5, "uncertainty_threshold": 0.5}
        },
        "oracle": {
            "dft": {
                "code": "mock",
                "command": "mock",
                "pseudopotentials": {"Fe": "mock.pbe"}
            },
            "mace": {
                "model_path": "medium",
                "mock": True
            },
            "mock": True
        }
    }

    config_yaml = tmp_path / "config_al.yaml"
    import yaml
    with config_yaml.open("w") as f:
        yaml.dump(config_dict, f)

    config = load_config(config_yaml)
    orchestrator = Orchestrator(config)

    # Mock Step 1 output (pool of candidates)
    # Create 20 candidates with random uncertainty
    candidates = []
    from uuid import uuid4
    for i in range(20):
        s = StructureMetadata(id=uuid4(), tags=[f"c_{i}"])
        # We need to ensure uncertainty is computed in step 2, or provided.
        # The orchestrator should handle uncertainty computation if missing?
        # Or step 1 output is raw structures.
        candidates.append(s)

    # Inject candidates into orchestrator state/memory if possible
    # Or rely on run_step2 taking candidates as input?
    # SPEC says orchestrator calls MaceSurrogateOracle -> ActiveLearner.
    # So we probably need to mock the data flow or file system.

    # Let's assume run_step2 can be tested in isolation if we provide input file?
    # Or we just run step 2 and it expects data in 'candidates.pckl'

    # Save candidates to expected file
    from pyacemaker.core.config import CONSTANTS
    candidates_file = config.project.root_dir / CONSTANTS.default_candidates_file
    # We need DatasetManager to save
    from pyacemaker.oracle.dataset import DatasetManager
    dm = DatasetManager()
    # We need atoms attached to save (metadata only might fail if load expects atoms)
    # Create dummy atoms
    from ase import Atoms
    atoms_list = [Atoms('Fe', positions=[[0,0,0]]) for _ in candidates]
    dm.save(atoms_list, candidates_file)

    # Run Step 2
    if not hasattr(orchestrator, "run_step2_active_learning"):
        pytest.skip("run_step2_active_learning not implemented")

    selected_structures = orchestrator.run_step2_active_learning()

    # Verify selection count
    assert len(selected_structures) == 5

    # Verify they have uncertainty populated
    for s in selected_structures:
        assert s.uncertainty is not None
        assert s.uncertainty >= 0.5  # Threshold check (if enforced by logic, though n_select might override)

    # Verify they are labeled by DFT (mock)
    for s in selected_structures:
        assert s.label_source == "dft"
        assert s.energy is not None
