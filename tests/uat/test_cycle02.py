"""UAT for Cycle 02: DIRECT Sampling & Active Learning."""

from pathlib import Path

import pytest

from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.orchestrator import Orchestrator


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
        "structure_generator": {
            "default_element": "Fe",
            "supercell": [2, 2, 2],
            "rattle_amplitude": 0.1
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

    config_yaml = tmp_path / "config.yaml"
    import yaml
    with config_yaml.open("w") as f:
        yaml.dump(config_dict, f)

    from pyacemaker.core.config_loader import load_config
    try:
        config = load_config(config_yaml)
    except Exception as e:
        pytest.fail(f"Failed to load config: {e}")

    orchestrator = Orchestrator(config)

    if not hasattr(orchestrator, "run_step1_direct_sampling"):
        pytest.skip("run_step1_direct_sampling not implemented")

    generated_structures = orchestrator.run_step1_direct_sampling()

    assert len(generated_structures) == 50
    for s in generated_structures:
        is_direct = s.generation_method == "direct" or "direct" in s.tags
        assert is_direct, f"Structure {s.id} missing direct tag/method"
        assert s.status == "NEW"

        # Verify content
        assert "atoms" in s.features
        atoms = s.features["atoms"]
        assert len(atoms) > 0
        assert "Fe" in atoms.get_chemical_symbols()


def test_cycle02_active_learning(tmp_path: Path) -> None:
    """Scenario 02: Active Learning Selection."""
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

    candidates = []
    from uuid import uuid4
    for i in range(20):
        s = StructureMetadata(id=uuid4(), tags=[f"c_{i}"])
        candidates.append(s)

    from pyacemaker.core.config import CONSTANTS
    candidates_file = config.project.root_dir / CONSTANTS.default_candidates_file
    from pyacemaker.oracle.dataset import DatasetManager
    dm = DatasetManager()
    from ase import Atoms
    atoms_list = [Atoms('Fe', positions=[[0,0,0]]) for _ in candidates]
    dm.save(atoms_list, candidates_file)

    if not hasattr(orchestrator, "run_step2_active_learning"):
        pytest.skip("run_step2_active_learning not implemented")

    selected_structures = orchestrator.run_step2_active_learning()

    assert len(selected_structures) == 5

    for s in selected_structures:
        # Check uncertainty via state or direct attribute if populated by mock
        assert s.uncertainty_state is not None
        assert s.uncertainty_state.gamma_max >= 0.5

    for s in selected_structures:
        assert s.label_source == "dft" # Or mace if using mace as labeler (but here oracle is DFT)
        # Check DFT results (energy/forces) are present
        assert s.energy is not None
        assert s.forces is not None
        assert len(s.forces) == len(s.features["atoms"])
