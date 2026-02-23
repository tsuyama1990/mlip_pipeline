"""Tests for streaming components."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.domain_models.models import (
    StructureMetadata,
)
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def streaming_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Mock config for streaming tests."""
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="streaming_test", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(pseudopotentials={"H": "H.upf"}), mock=True
        ),
        trainer=TrainerConfig(mock=True),
        structure_generator=StructureGeneratorConfig(strategy="random"),
    )


def test_trainer_train_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify Trainer.train consumes generator without materializing list."""
    trainer = PacemakerTrainer(streaming_config)

    # Create a generator that yields structures
    # We want to verify it is consumed lazily.
    # However, save_iter inside train consumes it.
    # We mock save_iter to verify it receives a generator.

    def structure_gen() -> Iterator[StructureMetadata]:
        for _ in range(10):
            yield StructureMetadata(
                features={"atoms": Atoms("H")},
                energy=-1.0,
                forces=[[0.0, 0.0, 0.0]],
            )

    with patch.object(trainer.dataset_manager, "save_iter") as mock_save:
        # Mock save_iter to consume the generator passed to it
        # This is crucial because trainer.train relies on consumption to count stats
        def consume_generator(data, *args, **kwargs):
            for _ in data:
                pass
        mock_save.side_effect = consume_generator

        trainer.train(structure_gen())

        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        data_arg = args[0]

        # Verify passed argument is an iterator/generator
        assert isinstance(data_arg, Iterator)
        assert not isinstance(data_arg, list)


def test_trainer_select_active_set_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify Trainer.select_active_set streaming behavior."""
    trainer = PacemakerTrainer(streaming_config)

    def structure_gen() -> Iterator[StructureMetadata]:
        for _ in range(10):
            yield StructureMetadata(features={"atoms": Atoms("H")})

    with (
        patch.object(trainer.dataset_manager, "save_iter") as mock_save,
        patch.object(trainer.dataset_manager, "load_iter") as mock_load,
    ):
        # Mock loading back the selected file to avoid errors
        # Since we mock save_iter, the file won't exist.
        # We also need to mock active_set_selector.select if not in mock mode,
        # but trainer config mock=True handles that path.

        # In mock mode, it reloads candidates and slices them.
        # We need to ensure load_iter returns something.
        mock_load.return_value = iter([Atoms("H")])  # Dummy return for extraction loop

        _ = trainer.select_active_set(structure_gen(), n_select=5)

        # Check save_iter called with generator for candidates
        # It's called twice in mock mode: once for candidates, once for selected.
        assert mock_save.call_count >= 1
        first_call_args = mock_save.call_args_list[0][0]
        assert isinstance(first_call_args[0], Iterator)


def test_orchestrator_training_phase_streaming(streaming_config: PYACEMAKERConfig) -> None:
    """Verify Orchestrator._run_training_phase uses streaming."""
    orchestrator = Orchestrator(config=streaming_config)

    # Mock trainer
    orchestrator.trainer = MagicMock()

    # Mock dataset manager load_iter to return iterator
    # We need to simulate that dataset exists
    # Create the directory first
    orchestrator.training_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator.training_path.touch()

    # Mock DatasetSplitter to return empty stream so we focus on step 2 (full train)
    with patch("pyacemaker.orchestrator.DatasetSplitter") as MockSplitter:
        instance = MockSplitter.return_value
        instance.train_stream.return_value = iter([]) # Empty new items
        instance.processed_count = 0

        orchestrator._run_training_phase()

        # Verify trainer.train called with iterator
        orchestrator.trainer.train.assert_called_once()
        args = orchestrator.trainer.train.call_args[0]
        assert isinstance(args[0], Iterator)
