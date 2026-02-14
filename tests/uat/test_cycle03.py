"""UAT for Cycle 03: Trainer & Potential Generation."""

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import PYACEMAKERConfig, TrainerConfig
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.modules.trainer import PacemakerTrainer as Trainer


class TestCycle03UAT:
    """UAT Scenarios for Trainer."""

    @pytest.fixture
    def trainer(self) -> Trainer:
        config = MagicMock(spec=PYACEMAKERConfig)
        config.trainer = TrainerConfig(
            cutoff=5.0,
            order=3,
            mock=False,  # We want to test the logic that calls subprocess
        )
        # We use mock=False in config, but patch subprocess in tests.
        return Trainer(config)

    @patch("subprocess.run")
    def test_scenario_01_training_execution(self, mock_run: MagicMock, trainer: Trainer) -> None:
        """Scenario 01: Training Execution."""
        # Create dataset
        # Need to mock save_iter to avoid file creation errors?
        # Actually save_iter writes to temp dir, which is fine.

        structure = StructureMetadata(
            features={"atoms": Atoms("Fe")},
            energy=-10.0,
            forces=[[0.0, 0.0, 0.0]],
        )
        dataset = [structure]

        mock_run.return_value = MagicMock(returncode=0)

        potential = trainer.train(dataset)

        assert potential.type == "PACE"
        # Verify pace_train was called with correct args
        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pace_train"
        assert "--cutoff" in cmd
        assert "5.0" in cmd

    @patch("subprocess.run")
    def test_scenario_02_active_set_selection(self, mock_run: MagicMock, trainer: Trainer) -> None:
        """Scenario 02: Active Set Selection."""
        candidates = [StructureMetadata(features={"atoms": Atoms("Fe")}) for _ in range(5)]

        mock_run.return_value = MagicMock(returncode=0)

        # Mock DatasetManager.load_iter to return atoms with UUIDs
        # because select_active_set loads the OUTPUT file which won't exist because subprocess is mocked.
        trainer.dataset_manager = MagicMock()

        # Mock load_iter to return atoms with matching UUIDs
        mock_atoms_iter = []
        for s in candidates[:2]:
            a = Atoms("Fe")
            a.info["uuid"] = str(s.id)
            mock_atoms_iter.append(a)
        trainer.dataset_manager.load_iter.return_value = mock_atoms_iter

        # Ensure save_iter creates file
        def consume_iterator(data, path):  # type: ignore[no-untyped-def]
            for _ in data:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        trainer.dataset_manager.save_iter.side_effect = consume_iterator

        active_set = trainer.select_active_set(candidates, n_select=2)

        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pace_activeset"
        assert "--select" in cmd
        assert "2" in cmd

        assert len(active_set.structure_ids) == 2
        assert active_set.structure_ids[0] == candidates[0].id

    @patch("subprocess.run")
    def test_scenario_03_delta_learning(self, mock_run: MagicMock) -> None:
        """Scenario 03: Delta Learning Configuration."""
        config = MagicMock(spec=PYACEMAKERConfig)
        config.trainer = TrainerConfig(delta_learning="zbl", mock=False)
        trainer = Trainer(config)
        trainer.dataset_manager = MagicMock()  # Mock to avoid file IO issues

        # Ensure save_iter consumes the iterator AND creates file (so exists checks pass)
        def consume_iterator(data, path):  # type: ignore[no-untyped-def]
            for _ in data:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        trainer.dataset_manager.save_iter.side_effect = consume_iterator

        structure = StructureMetadata(
            features={"atoms": Atoms("Fe")},
            energy=-10.0,
            forces=[[0.0, 0.0, 0.0]],
        )
        dataset = [structure]

        mock_run.return_value = MagicMock(returncode=0)

        # Mock file existence checks in wrapper
        with patch("pathlib.Path.exists", return_value=True):
            trainer.train(dataset)

        # Verify baseline file generated and passed
        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        # Check if --baseline argument is present
        assert "--baseline" in cmd
        # And verify baseline file name contains "zbl"
        baseline_idx = cmd.index("--baseline") + 1
        assert "zbl" in cmd[baseline_idx]
