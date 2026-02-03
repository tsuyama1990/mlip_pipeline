import logging
from unittest.mock import MagicMock

import pytest

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    SimulationConfig,
    TrainingConfig,
)
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.orchestrator import Orchestrator


@pytest.fixture
def mock_components():
    return {
        "explorer": MagicMock(spec=Explorer),
        "oracle": MagicMock(spec=Oracle),
        "trainer": MagicMock(spec=Trainer),
        "validator": MagicMock(spec=Validator)
    }


@pytest.fixture
def valid_config():
    return SimulationConfig(
        project_name="TestProject",
        dft=DFTConfig(code="qe", ecutwfc=40.0, kpoints=[2, 2, 2]),
        training=TrainingConfig(code="pacemaker", cutoff=5.0),
        exploration=ExplorationConfig()
    )


def test_orchestrator_initialization(valid_config, mock_components, caplog):
    """Test that Orchestrator initializes correctly and logs startup."""
    caplog.set_level(logging.INFO)

    orchestrator = Orchestrator(
        config=valid_config,
        explorer=mock_components["explorer"],
        oracle=mock_components["oracle"],
        trainer=mock_components["trainer"],
        validator=mock_components["validator"]
    )

    assert orchestrator.config == valid_config
    assert "PYACEMAKER initialized for project: TestProject" in caplog.text


def test_orchestrator_run_loop(valid_config, mock_components):
    """Test that run_loop executes the steps."""
    orchestrator = Orchestrator(
        config=valid_config,
        explorer=mock_components["explorer"],
        oracle=mock_components["oracle"],
        trainer=mock_components["trainer"],
        validator=mock_components["validator"]
    )

    # Mock return values
    mock_components["explorer"].explore.return_value = []
    mock_components["oracle"].compute.return_value = []

    orchestrator.run_loop()

    mock_components["explorer"].explore.assert_called()
    mock_components["oracle"].compute.assert_called()
    mock_components["trainer"].train.assert_called()
    mock_components["validator"].validate.assert_called()
