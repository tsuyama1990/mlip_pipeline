from pathlib import Path

import pytest
import yaml

# Imports that might fail
try:
    from mlip_autopipec.config.config_model import Config
    from mlip_autopipec.orchestration.orchestrator import Orchestrator
except ImportError:
    Orchestrator = None # type: ignore
    Config = None # type: ignore

@pytest.mark.skipif(Orchestrator is None, reason="Orchestrator not implemented yet")
def test_skeleton_loop_execution(temp_project_dir: Path, valid_config_yaml: Path, dummy_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Runs the full skeleton loop for 1 iteration.
    Uses PYACEMAKER_MOCK_MODE to simulate training.
    """
    # Set mock mode
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")

    # Load config
    with valid_config_yaml.open("r") as f:
        data = yaml.safe_load(f)
    config = Config.model_validate(data)

    # Initialize Orchestrator
    # We assume Orchestrator takes config and work_dir?
    # Spec says: Orchestrator(config: Config)
    # But where does it write state.json?
    # Probably in the CWD or we should pass a working directory.
    # The spec blueprints show `self.state.save()`, implies hardcoded path or configured.
    # UAT says "The user creates a folder project_test... runs command... sees state.json".
    # So it writes to CWD.
    # We must change CWD to temp_project_dir for the test.

    monkeypatch.chdir(temp_project_dir)

    orchestrator = Orchestrator(config)
    orchestrator.run()

    # Assertions
    state_file = temp_project_dir / "workflow_state.json" # Assuming name
    assert state_file.exists()

    import json
    with state_file.open("r") as f:
        state_data = json.load(f)

    assert state_data["iteration"] >= 1
    # Check output potential
    output_pot = temp_project_dir / "output_potential.yace" # Based on PacemakerTrainer mock
    assert output_pot.exists()
