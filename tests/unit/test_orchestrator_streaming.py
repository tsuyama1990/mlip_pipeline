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
from pyacemaker.domain_models.models import StructureMetadata
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
        orchestrator=OrchestratorConfig(
            validation_split=0.1, max_validation_size=10
        ),
    )


def test_cold_start_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify that cold start streams data and extends dataset correctly."""

    # Create an iterator that yields structures
    def structure_gen() -> Iterator[StructureMetadata]:
        for _ in range(5):
            # Must provide atoms for metadata_to_atoms
            yield StructureMetadata(
                features={"atoms": Atoms("H", positions=[[0, 0, 0]])}
            )

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
    for _ in orchestrator.dataset_manager.load_iter(orchestrator.dataset_path):
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
    """Verify validation splitting logic and bounds."""
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
    orchestrator.config.orchestrator.validation_split = 1.0
    orchestrator.config.orchestrator.max_validation_size = 10

    # Run _split_dataset_streams directly
    train_stream, val_list = orchestrator._split_dataset_streams()

    # Consume train_stream to trigger splitting (it's a generator)
    consumed_train = list(train_stream)

    # Validation list should be capped at 10
    assert len(val_list) == 10

    # Remaining items (100 - 10 = 90) should be in train stream
    assert len(consumed_train) == 90

    # Now test _run_validation_phase
    # We need to manually set _validation_set because we called _split directly
    orchestrator._validation_set = val_list

    orchestrator._run_validation_phase()

    assert mock_validator.validate.called
    call_args = mock_validator.validate.call_args
    test_set = call_args[0][1]

    # Explicitly check identity or content
    assert test_set is val_list
    assert len(test_set) == 10
