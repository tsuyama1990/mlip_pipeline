"""Tests for Orchestrator streaming and memory efficiency."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
)
from pyacemaker.core.dataset import DatasetSplitter
from pyacemaker.domain_models.models import (
    ActiveSet,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def streaming_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PYACEMAKERConfig:
    """Mock config."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="test", root_dir=tmp_path),
        oracle=OracleConfig(dft=DFTConfig(pseudopotentials={"H": "H.upf"})),
        structure_generator=StructureGeneratorConfig(strategy="random"),
        orchestrator=OrchestratorConfig(validation_split=0.1, max_validation_size=10),
    )


def test_cold_start_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify that cold start streams data and extends dataset correctly."""

    # Create an iterator that yields structures
    def structure_gen() -> Iterator[StructureMetadata]:
        for _ in range(5):
            # Must provide atoms for metadata_to_atoms
            yield StructureMetadata(features={"atoms": Atoms("H", positions=[[0, 0, 0]])})

    mock_gen = MagicMock()
    mock_gen.generate_initial_structures.return_value = structure_gen()

    mock_oracle = MagicMock()

    # oracle.compute_batch should return an iterator (or pass through)
    def compute_passthrough(structures: Iterator[StructureMetadata]) -> Iterator[StructureMetadata]:
        # Emulate lazy consumption
        yield from structures

    mock_oracle.compute_batch.side_effect = compute_passthrough

    orchestrator = Orchestrator(
        config=streaming_config,
        structure_generator=mock_gen,
        oracle=mock_oracle,
        trainer=MagicMock(),
        dynamics_engine=MagicMock(),
        validator=MagicMock(),
    )

    # Run cold start
    orchestrator._run_cold_start()

    # Verify dataset file exists and has 5 items
    assert orchestrator.dataset_path.exists()

    # Check count by reading file
    count = 0
    for _ in orchestrator.dataset_manager.load_iter(orchestrator.dataset_path, verify=False):
        count += 1
    assert count == 5

    # Verify generator was called
    mock_gen.generate_initial_structures.assert_called_once()
    # Verify compute_batch was called with the generator (not a list)
    mock_oracle.compute_batch.assert_called_once()
    args = mock_oracle.compute_batch.call_args[0][0]
    # Check that the argument passed to compute_batch was indeed the generator/iterator
    assert isinstance(args, Iterator)


def test_validation_slice(streaming_config: PYACEMAKERConfig) -> None:
    """Verify validation splitting logic and bounds using DatasetSplitter."""
    orchestrator = Orchestrator(config=streaming_config)

    # Fill dataset with 100 items
    dataset_path = orchestrator.dataset_path
    atoms_list = [Atoms("H") for _ in range(100)]
    orchestrator.dataset_manager.save_iter(iter(atoms_list), dataset_path)

    orchestrator.current_potential = MagicMock()

    mock_validator = MagicMock()
    mock_validator.validate.return_value = MagicMock(status="success")
    orchestrator.validator = mock_validator

    # Force split_ratio=1.0 to try to put everything in validation,
    # but max_validation_size=10 should limit it.
    validation_split = 1.0
    max_validation_size = 10

    # Manually test DatasetSplitter
    splitter = DatasetSplitter(
        dataset_path=orchestrator.dataset_path,
        validation_path=orchestrator.validation_path,
        dataset_manager=orchestrator.dataset_manager,
        validation_split=validation_split,
        max_validation_size=max_validation_size,
    )

    # Consume train_stream to trigger splitting
    # We must patch load_iter to disable verification inside DatasetSplitter
    from unittest.mock import patch

    with patch.object(
        orchestrator.dataset_manager,
        "load_iter",
        side_effect=lambda p, **kwargs: orchestrator.dataset_manager.__class__.load_iter(
            orchestrator.dataset_manager, p, verify=False, **kwargs
        ),
    ):
        train_stream = splitter.train_stream()
        consumed_train = list(train_stream)

    # Verify validation file exists and count items
    assert orchestrator.validation_path.exists()

    val_count = 0
    for _ in orchestrator.dataset_manager.load_iter(orchestrator.validation_path, verify=False):
        val_count += 1

    # Validation file should be capped at 10 (approx, since splitter flushes in batches,
    # but the logic caps self._val_count >= max. The exact number on disk matches _val_count).
    assert val_count == 10

    # Remaining items (100 - 10 = 90) should be in train stream
    assert len(consumed_train) == 90

    # Now test _run_validation_phase integration
    # It should load from file
    orchestrator._run_validation_phase()

    assert mock_validator.validate.called
    call_args = mock_validator.validate.call_args
    test_set = call_args[0][1]

    # Explicitly check content
    assert len(list(test_set)) == 10


def test_exploration_integration(streaming_config: PYACEMAKERConfig) -> None:
    """Test exploration phase integration."""
    orchestrator = Orchestrator(config=streaming_config)
    orchestrator.current_potential = MagicMock()  # Mock potential

    # Mock DynamicsEngine
    mock_dynamics = MagicMock()
    s = StructureMetadata()
    s.uncertainty_state = UncertaintyState(gamma_max=10.0)
    mock_dynamics.run_exploration.return_value = iter([s])
    orchestrator.dynamics_engine = mock_dynamics

    # Mock Generator
    mock_gen = MagicMock()
    mock_gen.generate_initial_structures.return_value = [s]  # seeds
    mock_gen.generate_batch_candidates.return_value = iter([s])  # Just return same structure
    orchestrator.structure_generator = mock_gen

    # Mock Trainer
    mock_trainer = MagicMock()
    mock_trainer.select_active_set.return_value = ActiveSet(
        structure_ids=[s.id], structures=[s], selection_criteria="test"
    )
    orchestrator.trainer = mock_trainer

    # Run
    selected = orchestrator._run_exploration_and_selection_phase()

    assert selected is not None
    assert len(selected) == 1
    assert selected[0] == s
    # Verify seeds were passed
    mock_dynamics.run_exploration.assert_called_once()
    args = mock_dynamics.run_exploration.call_args
    assert args[0][0] == orchestrator.current_potential
    assert list(args[0][1]) == [s]
