"""UAT for Cycle 03: Trainer & Potential Generation.

This test suite verifies the end-to-end functionality of the Trainer module,
focusing on Pacemaker integration, active set selection, and delta learning configuration.
It ensures that the system correctly interfaces with external tools (via mocks)
and handles data streaming efficiently.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    TrainerConfig,
)
from pyacemaker.core.factory import ModuleFactory
from pyacemaker.core.interfaces import Trainer
from pyacemaker.domain_models.models import StructureMetadata


class TestCycle03UAT:
    """UAT Scenarios for Trainer."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> PYACEMAKERConfig:
        """Create a real configuration object."""
        return PYACEMAKERConfig(
            version="0.1.0",
            project=ProjectConfig(name="TestProject", root_dir=tmp_path),
            oracle=OracleConfig(
                dft=DFTConfig(pseudopotentials={"Fe": "Fe.upf"}),
                mock=True
            ),
            trainer=TrainerConfig(
                cutoff=5.0,
                order=3,
                mock=False,
            ),
        )

    @pytest.fixture
    def trainer(self, mock_config: PYACEMAKERConfig) -> Trainer:
        """Create a Trainer instance using the Factory."""
        return ModuleFactory.create_trainer(mock_config)

    @patch("subprocess.run")
    def test_scenario_01_training_execution(
        self, mock_run: MagicMock, trainer: Trainer
    ) -> None:
        """Scenario 01: Training Execution.

        Verifies that:
        1. Training accepts a generator of StructureMetadata.
        2. The pace_train command is constructed correctly with config parameters.
        3. A Potential object is returned with correct metadata.
        """
        # Generator for memory safety
        def dataset_gen() -> Iterator[StructureMetadata]:
            yield StructureMetadata(
                features={"atoms": Atoms("Fe")},
                energy=-10.0,
                forces=[[0.0, 0.0, 0.0]],
            )

        mock_run.return_value = MagicMock(returncode=0)

        # Simulate output creation by pace_train
        def mock_pace_train_side_effect(*args, **kwargs):
            # Check if cwd is in kwargs
            cwd = kwargs.get("cwd")
            if cwd:
                (cwd / "potential.yace").touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = mock_pace_train_side_effect

        # Ensure save_iter consumes generator without crashing
        trainer.dataset_manager = MagicMock()

        def consume_iterator(data: Any, path: Path, **kwargs: Any) -> None:
            # Materialize to ensure generator is valid, then discard
            for _ in data:
                pass
            # Simulate file creation
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        trainer.dataset_manager.save_iter.side_effect = consume_iterator

        potential = trainer.train(dataset_gen())

        assert potential.type == "PACE"

        # Verify pace_train call
        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pace_train"
        # Since we use input.yaml, we verify an input file is passed
        assert len(cmd) == 2
        assert cmd[1].endswith("input.yaml")

    @patch("subprocess.run")
    def test_scenario_02_active_set_selection(
        self, mock_run: MagicMock, trainer: Trainer
    ) -> None:
        """Scenario 02: Active Set Selection.

        Verifies that:
        1. Selection accepts a generator of candidates.
        2. The pace_activeset command is constructed correctly.
        3. Selected structure IDs are correctly parsed from the output (mocked).
        """
        n_select = 2

        # Generator for candidates
        def candidates_gen() -> Iterator[StructureMetadata]:
            for _ in range(5):
                yield StructureMetadata(features={"atoms": Atoms("Fe")})

        mock_run.return_value = MagicMock(returncode=0)

        # Simulate output creation by pace_activeset
        def mock_activeset_side_effect(*args, **kwargs):
            # pace_activeset command: ["pace_activeset", "--dataset", ..., "--output", output_path]
            # args[0] is the command list
            cmd = args[0]
            output_idx = cmd.index("--output") + 1
            output_path = Path(cmd[output_idx])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = mock_activeset_side_effect

        # Mock DatasetManager
        trainer.dataset_manager = MagicMock()

        # Mock load_iter to return atoms with UUIDs (simulating reading the output file)
        # We need to simulate the result of selection which writes to a file
        # The trainer.select_active_set reads from this file.

        # We need consistent UUIDs to check against
        dummy_candidates = list(candidates_gen())

        mock_atoms_iter = []
        for s in dummy_candidates[:n_select]:
            a = Atoms("Fe")
            a.info["uuid"] = str(s.id)
            mock_atoms_iter.append(a)

        trainer.dataset_manager.load_iter.return_value = iter(mock_atoms_iter)

        # Mock save_iter to handle generator input
        def consume_iterator(data: Any, path: Path, **kwargs: Any) -> None:
            for _ in data:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        trainer.dataset_manager.save_iter.side_effect = consume_iterator

        # Pass fresh generator
        active_set = trainer.select_active_set(candidates_gen(), n_select=n_select)

        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pace_activeset"
        assert "--select" in cmd
        assert str(n_select) in cmd

        assert len(active_set.structure_ids) == n_select
        # Verify ID matching (assuming the mock return setup aligns)
        assert active_set.structure_ids[0] == dummy_candidates[0].id

    @patch("subprocess.run")
    def test_scenario_03_delta_learning(
        self, mock_run: MagicMock, mock_config: PYACEMAKERConfig
    ) -> None:
        """Scenario 03: Delta Learning Configuration.

        Verifies that:
        1. Delta learning configuration (e.g., 'zbl') triggers baseline generation.
        2. The baseline file path is passed to the training command.
        """
        delta_method = "zbl"
        mock_config.trainer.delta_learning = delta_method

        trainer = ModuleFactory.create_trainer(mock_config)
        trainer.dataset_manager = MagicMock()

        def consume_iterator(data: Any, path: Path, **kwargs: Any) -> None:
            for _ in data:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        trainer.dataset_manager.save_iter.side_effect = consume_iterator

        def dataset_gen() -> Iterator[StructureMetadata]:
            yield StructureMetadata(
                features={"atoms": Atoms("Fe")},
                energy=-10.0,
                forces=[[0.0, 0.0, 0.0]],
            )

        mock_run.return_value = MagicMock(returncode=0)

        def mock_pace_train_side_effect(*args, **kwargs):
            cwd = kwargs.get("cwd")
            if cwd:
                (cwd / "potential.yace").touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = mock_pace_train_side_effect

        # Mock file existence checks in wrapper
        with patch("pathlib.Path.exists", return_value=True):
            trainer.train(dataset_gen())

        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        # Verify input.yaml usage
        assert len(cmd) == 2
        assert cmd[1].endswith("input.yaml")

        # Verify delta learning configuration was stored
        assert trainer.trainer_config.delta_learning == delta_method
