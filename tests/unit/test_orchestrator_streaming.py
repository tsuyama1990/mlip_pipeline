"""Tests for Orchestrator logic and streaming."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    OracleConfig,
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
    )


def test_cold_start_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify that cold start streams data and extends dataset correctly."""

    # Create an iterator that yields structures
    from ase import Atoms
    def structure_gen() -> Iterator[StructureMetadata]:
        for _ in range(5):
            yield StructureMetadata(features={"atoms": Atoms("H")})

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

    # Verify dataset file was written
    assert orchestrator.dataset_path.exists()
    # Verify size (lazy check via loading, without materializing list)
    count = sum(1 for _ in orchestrator.dataset_manager.load_iter(orchestrator.dataset_path))
    assert count == 5

    # Verify generator was called
    mock_gen.generate_initial_structures.assert_called_once()
    # Verify compute_batch was called with the generator (not a list)
    mock_oracle.compute_batch.assert_called_once()
    args = mock_oracle.compute_batch.call_args[0][0]
    # Check that the argument passed to compute_batch was indeed the generator/iterator
    assert isinstance(args, Iterator)


def test_validation_split_and_gating(streaming_config: PYACEMAKERConfig) -> None:
    """Verify validation split logic and failure gating."""
    orchestrator = Orchestrator(config=streaming_config)
    # Prepare dataset file
    from ase import Atoms

    from pyacemaker.core.utils import metadata_to_atoms

    # Use real Atoms to satisfy validation
    data = [StructureMetadata(id=uuid4(), features={"atoms": Atoms("H")}) for _ in range(100)]

    orchestrator.dataset_manager.save_iter(
        (metadata_to_atoms(s) for s in data), orchestrator.dataset_path
    )

    orchestrator.current_potential = MagicMock()

    # Mock components
    orchestrator.trainer = MagicMock()
    orchestrator.validator = MagicMock()

    # 1. Run Training Phase -> Should create split
    orchestrator._run_training_phase()

    # Check split
    train_args_iter = orchestrator.trainer.train.call_args[0][0]
    train_args = list(train_args_iter) # consume

    # We used random split, so exact 90/10 isn't guaranteed but should be close
    # The config has default split 0.1? No, defaults.yaml has 0.1.
    # config object passed `streaming_config` uses `CONSTANTS` which load defaults.

    val_len = len(orchestrator._validation_set)
    train_len = len(train_args)
    total = val_len + train_len

    assert total == 100
    assert 0 < val_len < 30  # Should be around 10

    # Check that they are disjoint
    train_ids = {s.id for s in train_args}
    val_ids = {s.id for s in orchestrator._validation_set}
    assert train_ids.isdisjoint(val_ids)

    # 2. Run Validation Phase (Success)
    orchestrator.validator.validate.return_value = MagicMock(status="success")
    result = orchestrator._run_validation_phase()
    assert result is True

    # Verify validator received the validation set
    val_args = orchestrator.validator.validate.call_args[0][1]
    assert val_args == orchestrator._validation_set

    # 3. Run Validation Phase (Failure)
    orchestrator.validator.validate.return_value = MagicMock(status="failed")
    result = orchestrator._run_validation_phase()
    assert result is False
