import json
from pathlib import Path

import pytest
import yaml

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.orchestration.orchestrator import Orchestrator


def test_skeleton_loop(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")
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
    orch = Orchestrator(config)

    # Run
    orch.run()

    # Verify
    state_file = temp_dir / "state.json"
    assert state_file.exists()

    # Check iteration in state
    with state_file.open() as f:
        state = json.load(f)
    assert state["iteration"] == 1

    # Check output potential (MockTrainer should create it)
    # Orchestrator renames it to potential_iter_0.yace
    assert (temp_dir / "potential_iter_0.yace").exists()
