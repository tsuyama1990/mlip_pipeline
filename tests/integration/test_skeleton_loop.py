import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.factory import create_components
from mlip_autopipec.orchestration.mocks import MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator


def test_skeleton_loop(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")
    monkeypatch.chdir(temp_dir)  # Run in temp dir

    # Create a valid XYZ file
    data_file = temp_dir / "data.xyz"
    with data_file.open("w") as f:
        f.write("2\n")
        f.write(
            'Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" Properties=species:S:1:pos:R:3:forces:R:3 energy=-10.0\n'
        )
        f.write("Si 0.0 0.0 0.0 0.0 0.0 0.0\n")
        f.write("Si 1.35 1.35 1.35 0.0 0.0 0.0\n")

    config_data = {
        "project": {"name": "IntegrationTest"},
        "training": {"dataset_path": str(data_file), "max_epochs": 10},
        "orchestrator": {"max_iterations": 1},
        "exploration": {"strategy": "mock"},  # Force mock explorer via factory
        "selection": {"method": "mock"},      # Force mock selector via factory
        "oracle": {"method": "mock"},         # Force mock oracle via factory
        "validation": {"run_validation": True},
        "dft": {"pseudopotentials": {"Si": "Si.upf"}},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    # Initialize components using the Factory
    # We patch ValidationRunner to use MockValidator because we don't have LAMMPS/ASE-LAMMPS in this env
    with patch("mlip_autopipec.factory.ValidationRunner", side_effect=lambda _: MockValidator()):
        explorer, selector, oracle, trainer, validator = create_components(config)

    orch = Orchestrator(
        config=config,
        explorer=explorer,
        selector=selector,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
    )

    # Run
    orch.run()

    # Verify
    state_file = temp_dir / "state.json"
    assert state_file.exists()

    # Check iteration in state
    with state_file.open() as f:
        state = json.load(f)
    assert state["iteration"] == 1

    # Check history
    assert len(state["history"]) == 1
    history_entry = state["history"][0]
    assert history_entry["status"] == "success"
    assert "potential_path" in history_entry
    # Ensure strict schema compliance
    assert "potential" not in history_entry

    # Check output potential (MockTrainer should create it in active_learning/iter_000/training/output_potential.yace)
    # And copy to potentials/generation_000.yace

    assert (temp_dir / "potentials/generation_000.yace").exists()
