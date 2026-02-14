"""Tests for Pacemaker Wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.trainer.wrapper import PacemakerWrapper


class TestPacemakerWrapper:
    """Tests for PacemakerWrapper."""

    @pytest.fixture
    def wrapper(self) -> PacemakerWrapper:
        return PacemakerWrapper()

    @patch("subprocess.run")
    def test_train_command_construction(
        self, mock_run: MagicMock, wrapper: PacemakerWrapper
    ) -> None:
        """Test that pace_train command is constructed correctly."""
        mock_run.return_value = MagicMock(returncode=0)

        dataset_path = Path("data.pckl.gzip")
        output_dir = Path("output")
        params = {
            "cutoff": 6.0,
            "order": 2,
            "max_epochs": 100,
            "batch_size": 32,
        }

        wrapper.train(dataset_path, output_dir, params)

        mock_run.assert_called_once()
        # args[0] is the command list
        cmd = mock_run.call_args[0][0]

        assert cmd[0] == "pace_train"
        assert "--dataset" in cmd
        assert str(dataset_path) in cmd
        assert "--output-dir" in cmd
        assert str(output_dir) in cmd
        assert "--cutoff" in cmd
        assert "6.0" in cmd
        assert "--max-epochs" in cmd
        assert "100" in cmd
        assert "--batch-size" in cmd
        assert "32" in cmd

    @patch("subprocess.run")
    def test_train_with_initial_potential(
        self, mock_run: MagicMock, wrapper: PacemakerWrapper
    ) -> None:
        """Test training with initial potential."""
        mock_run.return_value = MagicMock(returncode=0)
        initial_pot = Path("initial.yace")

        wrapper.train(Path("data.pckl.gzip"), Path("output"), {}, initial_potential=initial_pot)

        cmd = mock_run.call_args[0][0]
        assert "--initial-potential" in cmd
        assert str(initial_pot) in cmd

    @patch("subprocess.run")
    def test_activeset_command_construction(
        self, mock_run: MagicMock, wrapper: PacemakerWrapper
    ) -> None:
        """Test that pace_activeset command is constructed correctly."""
        mock_run.return_value = MagicMock(returncode=0)

        candidates_path = Path("candidates.pckl.gzip")
        num_select = 50
        output_path = Path("selected.pckl.gzip")

        wrapper.select_active_set(candidates_path, num_select, output_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert cmd[0] == "pace_activeset"
        assert "--dataset" in cmd
        assert str(candidates_path) in cmd
        assert "--select" in cmd
        assert str(num_select) in cmd
        assert "--output" in cmd
        assert str(output_path) in cmd
