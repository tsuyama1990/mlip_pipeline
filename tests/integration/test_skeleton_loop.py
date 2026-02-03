import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.orchestration.interfaces import Selector
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer


class MockSelector(Selector):
    def select(
        self,
        candidates: list[CandidateStructure],
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        return candidates  # Pass all


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
        "validation": {"run_validation": True},
        "dft": {"pseudopotentials": {"Si": "Si.upf"}},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    # Initialize components
    explorer = MockExplorer()
    selector = MockSelector()
    oracle = MockOracle()
    trainer = PacemakerTrainer(config.training)
    validator = MockValidator()

    orch = Orchestrator(config, explorer, selector, oracle, trainer, validator)

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


def test_skeleton_loop_adaptive(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")
    monkeypatch.chdir(temp_dir)

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
        "project": {"name": "IntegrationTestAdaptive"},
        "training": {"dataset_path": str(data_file), "max_epochs": 10},
        "orchestrator": {"max_iterations": 1},
        "exploration": {"strategy": "adaptive"},
        "validation": {"run_validation": True},
        "dft": {"pseudopotentials": {"Si": "Si.upf"}},
    }
    config_file = temp_dir / "config_adaptive.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    # Mock OTF Loop
    mock_otf = MagicMock()
    # execute_task returns a list of CandidateStructure
    mock_otf.execute_task.return_value = []

    # Initialize components
    explorer = AdaptiveExplorer(config, otf_loop=mock_otf)
    selector = MockSelector()
    oracle = MockOracle()
    trainer = PacemakerTrainer(config.training)
    validator = MockValidator()

    orch = Orchestrator(config, explorer, selector, oracle, trainer, validator)

    # Run
    orch.run()

    # Verify
    state_file = temp_dir / "state.json"
    assert state_file.exists()

    # Check that AdaptiveExplorer actually produced candidates (via Strain/Defect generator)
    # Since AdaptivePolicy creates StaticTasks, they run even if mock_otf is just a mock for MD
    with state_file.open() as f:
        state = json.load(f)

    # We expect some candidates from Static/Strain/Defect tasks
    assert state["history"][0]["candidates_count"] > 0
