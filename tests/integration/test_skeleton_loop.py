import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.orchestration.orchestrator import Orchestrator


class SimpleMockTrainer:
    def train(self, dataset: Path, previous_potential: Path | None, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        p = output_dir / "potential.yace"
        p.touch()
        return p


class SimpleMockExplorer:
    def explore(self, potential: Potential | None, work_dir: Path) -> dict[str, Any]:
        # Simulate finding candidates
        candidates_dir = work_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        (candidates_dir / "cand_1.xyz").touch()
        return {"halted": True, "candidates": [candidates_dir / "cand_1.xyz"]}


class SimpleMockOracle:
    def compute(self, structures: list[Path], work_dir: Path) -> list[Path]:
        # Simulate producing data
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "new_data.pckl"
        data_file.touch()
        # Return list of paths
        return [data_file]


class SimpleMockValidator:
    def validate(self, potential: Path) -> dict[str, Any]:
        return {"passed": True}


def test_skeleton_loop(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    monkeypatch.chdir(temp_dir)  # Run in temp dir

    data_file = temp_dir / "data.pckl"
    data_file.touch()

    config_data = {
        "project": {"name": "IntegrationTest"},
        "training": {"dataset_path": str(data_file), "max_epochs": 10},
        "orchestrator": {"max_iterations": 1},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    # Inject dependencies
    orch = Orchestrator(
        config=config,
        explorer=SimpleMockExplorer(),
        oracle=SimpleMockOracle(),
        trainer=SimpleMockTrainer(),
        validator=SimpleMockValidator(),
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

    # Check output potential
    # Orchestrator should rename/copy the output from trainer to a versioned file
    assert (temp_dir / "potential_iter_0.yace").exists()
