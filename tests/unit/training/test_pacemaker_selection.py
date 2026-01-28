import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper


@pytest.fixture
def training_config():
    return TrainingConfig(
        cutoff=6.0,
        b_basis_size=300,
        kappa=0.5,
        kappa_f=0.5,
        batch_size=32,
        template_path=Path("template.yaml")
    )

@pytest.fixture
def wrapper(training_config, tmp_path):
    return PacemakerWrapper(training_config, tmp_path)

def test_select_active_set_success(wrapper):
    """
    Test that select_active_set builds the correct command and parses indices.
    """
    candidates_path = Path("candidates.xyz")
    potential_path = Path("current.yace")

    # Mock subprocess.run
    with patch("subprocess.run") as mock_run, \
         patch("shutil.which", return_value="/usr/bin/pace_activeset"):

        # Simulate stdout output.
        # We assume the output is a list of indices, one per line, or comma separated?
        # Let's design the logic to expect space-separated indices on the last line.
        mock_run.return_value.stdout = "Selected structure indices:\n0 5 10"
        mock_run.return_value.returncode = 0

        # Mock Path.exists to pass validation
        with patch.object(Path, "exists", return_value=True):
            indices = wrapper.select_active_set(candidates_path, potential_path)

        # Verify the command called
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # Check command construction
        assert str(candidates_path) in cmd
        assert str(potential_path) in cmd
        assert "-d" in cmd
        assert "-p" in cmd

        # Verify parsing result
        # Since we haven't implemented parsing yet, this assertion defines our expectation for the implementation.
        assert indices == [0, 5, 10]

def test_select_active_set_failure(wrapper):
    """
    Test failure handling when pace_activeset crashes.
    """
    candidates_path = Path("candidates.xyz")
    potential_path = Path("current.yace")

    with patch("subprocess.run") as mock_run, \
         patch("shutil.which", return_value="/usr/bin/pace_activeset"):

        mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"])

        with patch.object(Path, "exists", return_value=True):
             indices = wrapper.select_active_set(candidates_path, potential_path)

        assert indices == []
