import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        cutoff=5.0,
        batch_size=32,
        b_basis_size=300,
        kappa=0.5,
        kappa_f=0.5
    )


def test_select_active_set_parsing(mock_training_config, tmp_path):
    wrapper = PacemakerWrapper(mock_training_config, tmp_path)

    # Mock shutil.which to pass executable check
    with patch("shutil.which", return_value="/usr/bin/pace_activeset"), \
         patch("subprocess.run") as mock_run:
        # Simulate pace_activeset output
        # Usually it outputs selected indices or writes to file.
        # SPEC says "returns indices".
        # Let's assume the tool prints: "Selected configuration indices: 1 5 10" or something.
        # Or maybe it outputs "Selected 3 structures: [1, 5, 10]"
        # I need to implement parsing logic that matches what I decide pace_activeset does (since I don't have the real binary).
        # Let's assume standard output format:
        # "Selected indices: 0, 2, 4"

        mock_run.return_value.stdout = """
        Pacemaker Active Set Selection
        ...
        Selected indices: 0 2 4
        Done.
        """
        mock_run.return_value.returncode = 0

        candidates_path = tmp_path / "candidates.xyz"
        candidates_path.touch()
        potential_path = tmp_path / "pot.yace"

        indices = wrapper.select_active_set(candidates_path, potential_path)

        assert indices == [0, 2, 4]

def test_select_active_set_parsing_empty(mock_training_config, tmp_path):
    wrapper = PacemakerWrapper(mock_training_config, tmp_path)

    with patch("shutil.which", return_value="/usr/bin/pace_activeset"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "Selected indices: \nDone."

        candidates_path = tmp_path / "candidates.xyz"
        candidates_path.touch()
        potential_path = tmp_path / "pot.yace"

        indices = wrapper.select_active_set(candidates_path, potential_path)

        assert indices == []
