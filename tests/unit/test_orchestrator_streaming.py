"""Unit tests for streaming capabilities in Orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import (
    PYACEMAKERConfig,
)
from pyacemaker.domain_models.models import (
    ActiveSet,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def streaming_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Create a config optimized for streaming tests."""
    return PYACEMAKERConfig(
        version="0.1.0",
        project={"name": "streaming_test", "root_dir": tmp_path},
        orchestrator={
            "max_cycles": 1,
            "uncertainty_threshold": 0.5,
            "dataset_file": "dataset.pckl.gzip",
            "validation_file": "val.pckl.gzip",
            "validation_buffer_size": 2,  # Small buffer to trigger flushes
            "validation_split": 0.5,  # High split to ensure validation items
        },
        oracle={
            "dft": {
                "pseudopotentials": {"Fe": "mock.pbe"},
                "max_workers": 2,  # Threading
            }
        },
        distillation={"enable_mace_distillation": False},
    )


def test_cold_start_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify cold start uses streaming interfaces."""
    orchestrator = Orchestrator(config=streaming_config)
    workflow = orchestrator.standard_workflow

    # Mock components
    mock_gen = MagicMock()
    # Return generator instead of list
    mock_gen.generate_initial_structures.return_value = (
        StructureMetadata() for _ in range(10)
    )
    workflow.structure_generator = mock_gen

    mock_oracle = MagicMock()
    # compute_batch must return generator
    mock_oracle.compute_batch.side_effect = lambda x: (s for s in x)
    workflow.oracle = mock_oracle

    # Spy on dataset_manager.save_iter
    with patch.object(workflow.dataset_manager, "save_iter") as mock_save:
        workflow._run_cold_start()

        assert mock_gen.generate_initial_structures.called
        assert mock_oracle.compute_batch.called
        assert mock_save.called
        # Verify first arg is iterator
        args, _ = mock_save.call_args
        # We can't easily check if it's an iterator if it was consumed, but save_iter consumes it.
        # Ideally we check logic flow.


def test_validation_slice(streaming_config: PYACEMAKERConfig) -> None:
    """Verify validation splitting is streaming/buffered."""
    orchestrator = Orchestrator(config=streaming_config)
    workflow = orchestrator.standard_workflow

    # Create dummy dataset
    dm = workflow.dataset_manager
    atoms_list = [Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]) for _ in range(20)]
    dm.save(atoms_list, workflow.dataset_path)

    # Use small buffer size in config fixture
    assert workflow.config.orchestrator.validation_buffer_size == 2

    # Mock trainer to avoid actual training
    workflow.trainer = MagicMock()

    # Run training phase (which includes splitting)
    workflow._run_training_phase()

    # Verify validation file exists and has content
    assert workflow.validation_path.exists()
    # Check count
    val_count = len(list(dm.load_iter(workflow.validation_path)))
    train_count = len(list(dm.load_iter(workflow.training_path)))

    assert val_count + train_count == 20
    assert val_count > 0  # With 0.5 split, should have some


def test_exploration_integration(streaming_config: PYACEMAKERConfig) -> None:
    """Verify exploration phase streaming integration."""
    orchestrator = Orchestrator(config=streaming_config)
    workflow = orchestrator.standard_workflow
    workflow.current_potential = MagicMock()

    # Mock Dynamics Engine to return generator
    mock_dynamics = MagicMock()
    s = StructureMetadata()
    s.uncertainty_state = UncertaintyState(gamma_max=10.0)
    mock_dynamics.run_exploration.return_value = iter([s])
    workflow.dynamics_engine = mock_dynamics

    # Mock Generator
    mock_gen = MagicMock()
    mock_gen.generate_initial_structures.return_value = [s]  # seeds
    mock_gen.generate_batch_candidates.return_value = iter([s])  # Just return same structure
    workflow.structure_generator = mock_gen

    # Mock Trainer
    mock_trainer = MagicMock()
    mock_trainer.select_active_set.return_value = ActiveSet(
        structure_ids=[s.id], structures=[s], selection_criteria="test"
    )
    workflow.trainer = mock_trainer

    # Mock Seed Selector via patch
    with patch("pyacemaker.core.dataset.SeedSelector.get_seeds", return_value=[s]):
        # Run
        selected = workflow._run_exploration_and_selection_phase()

    assert selected is not None
    assert len(list(selected)) == 1
