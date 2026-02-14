"""Tests for Active Set Selection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.trainer.active_set import ActiveSetSelector


class TestActiveSetSelector:
    """Tests for ActiveSetSelector."""

    @pytest.fixture
    def selector(self) -> ActiveSetSelector:
        return ActiveSetSelector()

    @patch("subprocess.run")
    def test_select_active_set(self, mock_run: MagicMock, selector: ActiveSetSelector) -> None:
        """Test selection of active set."""
        mock_run.return_value = MagicMock(returncode=0)

        candidates_path = Path("candidates.pckl.gzip")
        num_select = 100

        result_path = selector.select(candidates_path, num_select)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert cmd[0] == "pace_activeset"
        assert str(candidates_path) in cmd
        assert str(num_select) in cmd

        assert isinstance(result_path, Path)
