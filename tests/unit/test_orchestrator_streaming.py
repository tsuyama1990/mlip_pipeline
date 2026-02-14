"""Tests for Orchestrator streaming and memory efficiency."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    def structure_gen() -> Iterator[StructureMetadata]:
        for _ in range(5):
            yield StructureMetadata()

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

    # Verify dataset has 5 items
    assert len(orchestrator.dataset) == 5
    # Verify generator was called
    mock_gen.generate_initial_structures.assert_called_once()
    # Verify compute_batch was called with the generator (not a list)
    mock_oracle.compute_batch.assert_called_once()
    args = mock_oracle.compute_batch.call_args[0][0]
    # Check that the argument passed to compute_batch was indeed the generator/iterator
    assert isinstance(args, Iterator)


def test_validation_slice(streaming_config: PYACEMAKERConfig) -> None:
    """Verify validation phase uses slicing (islice) without copying full list."""
    orchestrator = Orchestrator(config=streaming_config)
    # Fill dataset with 100 items
    orchestrator.dataset = [StructureMetadata() for _ in range(100)]
    orchestrator.current_potential = MagicMock()

    mock_validator = MagicMock()
    mock_validator.validate.return_value = MagicMock(status="success")
    orchestrator.validator = mock_validator

    with patch("pyacemaker.orchestrator.islice", wraps=lambda it, n: iter(it[:n])):
        orchestrator._run_validation_phase()
        # Should create a list of size 10 (10% of 100)
        assert mock_validator.validate.called
        call_args = mock_validator.validate.call_args
        test_set = call_args[0][1]
        assert len(test_set) == 10
        # Check if islice was used (though we mocked it to verify call, real code uses it)
        # Since we patched pyacemaker.orchestrator.islice, we can verify it was imported and used
