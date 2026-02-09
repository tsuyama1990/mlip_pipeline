from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig, OrchestratorConfig
from mlip_autopipec.domain_models.results import ValidationMetrics
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def mock_config(tmp_path):
    # Minimal config for mocks
    config = MagicMock(spec=GlobalConfig)
    config.workdir = tmp_path
    config.max_cycles = 5
    config.orchestrator = OrchestratorConfig()

    # Explicitly set optional fields to None or Mock to avoid AttributeError with spec
    config.physics_baseline = None

    # Components
    # Avoid strict spec on ComponentsConfig to prevent issues with Pydantic fields
    comps = MagicMock()
    config.components = comps

    # Generator config
    comps.generator = MagicMock()
    comps.generator.n_structures = 10

    return config


@patch("mlip_autopipec.core.orchestrator.Dataset")
@patch("mlip_autopipec.core.orchestrator.StateManager")
@patch("mlip_autopipec.core.orchestrator.ComponentFactory")
def test_orchestrator_halt_logic(mock_factory, mock_state_mgr, mock_dataset_cls, mock_config):
    # Setup Mocks
    orchestrator = Orchestrator(mock_config)
    orchestrator.generator = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.validator = MagicMock()
    orchestrator.dataset = MagicMock()

    orchestrator.state_manager.state.current_cycle = 1

    # Setup Generator
    orchestrator.generator.generate.return_value = iter([MagicMock()])

    # Setup Dynamics to return structure with halt
    halted_structure = Structure(
        positions=np.zeros((1, 3)), atomic_numbers=np.array([1]), cell=np.eye(3), pbc=np.ones(3),
        tags={"provenance": "halt"}
    )
    orchestrator.dynamics.explore.return_value = iter([halted_structure])

    # Setup Trainer
    orchestrator.trainer.select_active_set.return_value = [halted_structure]
    orchestrator.trainer.train.return_value = MagicMock() # potential

    # Setup Oracle
    def mock_compute(structures):
        # Consume iterator to trigger side effects (halt counting)
        list(structures)
        return iter([halted_structure])

    orchestrator.oracle.compute.side_effect = mock_compute

    # Pre-set current potential to avoid loading from disk
    orchestrator.current_potential = MagicMock()
    orchestrator.current_potential.path = Path("mock_pot.yace")
    orchestrator.current_potential.metrics = {}

    # Run Cycle
    orchestrator._run_cycle()

    # Assertions
    # Since halt occurred, validation should NOT be called (or if logic changes, maybe it is called but ignored?)
    # Wait, spec says: "If Converged (no halts) -> Proceed to Validation".
    # So if halts occur, validation is skipped.
    orchestrator.validator.validate.assert_not_called()

    # Check if dataset appended
    orchestrator.dataset.append.assert_called()


@patch("mlip_autopipec.core.orchestrator.Dataset")
@patch("mlip_autopipec.core.orchestrator.StateManager")
@patch("mlip_autopipec.core.orchestrator.ComponentFactory")
def test_orchestrator_converged_validation_pass(mock_factory, mock_state_mgr, mock_dataset_cls, mock_config):
    # Setup Mocks
    orchestrator = Orchestrator(mock_config)
    orchestrator.generator = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.validator = MagicMock()
    orchestrator.dataset = MagicMock()

    orchestrator.state_manager.state.current_cycle = 1

    # Setup Generator
    orchestrator.generator.generate.return_value = iter([MagicMock()])

    # Setup Dynamics to return structure WITHOUT halt
    clean_structure = Structure(
        positions=np.zeros((1, 3)), atomic_numbers=np.array([1]), cell=np.eye(3), pbc=np.ones(3),
        tags={"provenance": "dynamics"}
    )
    orchestrator.dynamics.explore.return_value = iter([clean_structure])

    # Setup Oracle
    orchestrator.oracle.compute.return_value = iter([clean_structure])

    # Setup Trainer
    potential = MagicMock()
    potential.metrics = {}
    orchestrator.trainer.train.return_value = potential

    # Setup Validator to PASS
    metrics = ValidationMetrics(passed=True, phonon_stable=True)
    orchestrator.validator.validate.return_value = metrics

    # Pre-set current potential
    orchestrator.current_potential = MagicMock()
    orchestrator.current_potential.path = Path("mock_pot.yace")
    orchestrator.current_potential.metrics = {}

    # Run Cycle
    # This should trigger a StopIteration or set status to CONVERGED
    # But _run_cycle just runs one cycle. The loop is in run().
    # We test _run_cycle behavior.

    try:
        orchestrator._run_cycle()
    except StopIteration:
        pass # If we implement stopping by raising StopIteration or similar.
             # Or we check status update.

    # Assertions
    orchestrator.validator.validate.assert_called_once()
    orchestrator.state_manager.update_status.assert_called_with("CONVERGED")


@patch("mlip_autopipec.core.orchestrator.Dataset")
@patch("mlip_autopipec.core.orchestrator.StateManager")
@patch("mlip_autopipec.core.orchestrator.ComponentFactory")
def test_orchestrator_converged_validation_fail(mock_factory, mock_state_mgr, mock_dataset_cls, mock_config):
    # Setup Mocks
    orchestrator = Orchestrator(mock_config)
    orchestrator.generator = MagicMock()
    orchestrator.dynamics = MagicMock()
    orchestrator.oracle = MagicMock()
    orchestrator.trainer = MagicMock()
    orchestrator.validator = MagicMock()
    orchestrator.dataset = MagicMock()

    orchestrator.state_manager.state.current_cycle = 1

    # Setup Generator & Dynamics (No halts)
    clean_structure = Structure(
        positions=np.zeros((1, 3)), atomic_numbers=np.array([1]), cell=np.eye(3), pbc=np.ones(3),
        tags={"provenance": "dynamics"}
    )
    orchestrator.generator.generate.return_value = iter([MagicMock()])
    orchestrator.dynamics.explore.return_value = iter([clean_structure])
    orchestrator.oracle.compute.return_value = iter([clean_structure])

    # Setup Trainer
    potential = MagicMock()
    potential.metrics = {}
    orchestrator.trainer.train.return_value = potential

    # Setup Validator to FAIL
    failed_struct = Structure(
        positions=np.zeros((1, 3)), atomic_numbers=np.array([1]), cell=np.eye(3), pbc=np.ones(3),
        tags={"provenance": "validation_fail"}
    )
    metrics = ValidationMetrics(passed=False, failed_structures=[failed_struct])
    orchestrator.validator.validate.return_value = metrics

    # Pre-set current potential
    orchestrator.current_potential = MagicMock()
    orchestrator.current_potential.path = Path("mock_pot.yace")
    orchestrator.current_potential.metrics = {}

    # Run Cycle
    orchestrator._run_cycle()

    # Assertions
    orchestrator.validator.validate.assert_called_once()
    # Should NOT be converged
    assert call("CONVERGED") not in orchestrator.state_manager.update_status.call_args_list

    # Should append failed structures to dataset
    # The failed structures need to go through Oracle first? Yes.
    # Check if oracle.compute called twice? Once for dynamics structures, once for validation failures.
    assert orchestrator.oracle.compute.call_count >= 2
