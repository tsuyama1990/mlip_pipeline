import logging
from unittest.mock import MagicMock

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.orchestrator import Orchestrator


def test_skeleton_loop_execution(caplog):
    """
    Test the skeleton loop execution.
    This simulates a full cycle with mocked components.
    """
    caplog.set_level(logging.INFO)

    config_data = {
        "project_name": "SkeletonLoopTest",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0,
            "max_generations": 5
        },
        "exploration": {
            "strategy": "random",
            "max_temperature": 500.0,
            "steps": 10
        }
    }

    config = SimulationConfig(**config_data)

    # Mock Components
    explorer = MagicMock(spec=Explorer)
    oracle = MagicMock(spec=Oracle)
    trainer = MagicMock(spec=Trainer)
    validator = MagicMock(spec=Validator)

    # Set return values for mocks to allow the loop to proceed
    explorer.explore.return_value = []
    oracle.compute.return_value = []
    trainer.train.return_value = "mock_potential.yace"

    # Instantiate Orchestrator
    orchestrator = Orchestrator(
        config=config,
        explorer=explorer,
        oracle=oracle,
        trainer=trainer,
        validator=validator
    )

    # Run loop
    orchestrator.run_loop()

    # Assertions
    assert "Starting loop for project: SkeletonLoopTest" in caplog.text
    assert "Exploration completed." in caplog.text
    assert "DFT calculations completed." in caplog.text
    assert "Training completed." in caplog.text
    assert "Validation completed." in caplog.text

    explorer.explore.assert_called()
    oracle.compute.assert_called()
    trainer.train.assert_called()
    validator.validate.assert_called()
