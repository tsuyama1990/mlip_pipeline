import pytest
from unittest.mock import MagicMock, patch
from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator

def test_orchestrator_initialization() -> None:
    config_data = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    config = SimulationConfig(**config_data) # type: ignore[arg-type]
    orchestrator = Orchestrator(config)

    assert orchestrator.config.project_name == "TestProject"
    # Check that mocks are initialized
    assert orchestrator.explorer is not None
    assert orchestrator.oracle is not None
    assert orchestrator.trainer is not None
    assert orchestrator.validator is not None

def test_orchestrator_run_calls_components() -> None:
    config_data = {
        "project_name": "TestProject",
        "dft": {"code": "qe", "ecutwfc": 40.0, "kpoints": [1,1,1]},
        "training": {"code": "pacemaker", "cutoff": 5.0}
    }
    config = SimulationConfig(**config_data) # type: ignore[arg-type]
    orchestrator = Orchestrator(config)

    # Mock the internal components to verify calls
    orchestrator.explorer = MagicMock() # type: ignore[assignment]
    orchestrator.oracle = MagicMock() # type: ignore[assignment]
    orchestrator.trainer = MagicMock() # type: ignore[assignment]
    orchestrator.validator = MagicMock() # type: ignore[assignment]

    # Setup returns
    orchestrator.explorer.explore.return_value = ["struct1"]
    orchestrator.oracle.compute.return_value = ["labelled1"]
    orchestrator.trainer.train.return_value = "potential.yace"
    orchestrator.validator.validate.return_value.passed = True

    orchestrator.run()

    orchestrator.explorer.explore.assert_called_once()
    orchestrator.oracle.compute.assert_called_once_with(["struct1"])
    orchestrator.trainer.train.assert_called_once_with(["labelled1"])
    orchestrator.validator.validate.assert_called_once_with("potential.yace")

def test_orchestrator_failure() -> None:
    config_data = {
        "project_name": "TestProject",
        "dft": {"code": "qe", "ecutwfc": 40.0, "kpoints": [1,1,1]},
        "training": {"code": "pacemaker", "cutoff": 5.0}
    }
    config = SimulationConfig(**config_data) # type: ignore[arg-type]
    orchestrator = Orchestrator(config)
    orchestrator.explorer = MagicMock() # type: ignore[assignment]

    # Simulate error
    orchestrator.explorer.explore.side_effect = RuntimeError("Something went wrong")

    with pytest.raises(RuntimeError):
        orchestrator.run()
