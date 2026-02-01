from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.domain_models.config import PotentialConfig, ACEConfig
from mlip_autopipec.domain_models.job import JobStatus
import subprocess
import yaml

@pytest.fixture
def mock_train_config():
    return TrainingConfig(
        batch_size=10,
        max_epochs=1,
        active_set_optimization=True
    )

@pytest.fixture
def mock_pot_config():
    return PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        ace_params=ACEConfig(
            npot="FinnisSinclair",
            fs_parameters=[1, 1, 1, 0.5],
            ndensity=2
        )
    )

@pytest.fixture
def runner(tmp_path, mock_train_config, mock_pot_config):
    return PacemakerRunner(tmp_path, mock_train_config, mock_pot_config)

def test_train_success(runner):
    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.read_text") as mock_read:

        mock_run.return_value = MagicMock(returncode=0)
        mock_read.return_value = "RMSE Energy: 0.001\nRMSE Force: 0.02"

        # Mock existence of potential file
        # But we want to verify input.yaml creation, which is real.

        # We need to mock existence of output potential specifically
        # But allow input.yaml to be written?
        # The runner calls _generate_input_yaml which writes to file.

        # If we assume 'pace_train' doesn't run (mocked), no output file is created.
        # But we can check if input.yaml exists.

        with patch("pathlib.Path.exists") as mock_exists:
            # We need to handle checking specific paths.
            # side_effect can be a function.
            def exists_side_effect(self_path=None):
                # When calling Path("...").exists(), self is the path.
                # But here mock_exists is patching the unbound method? Or bound?
                # Usually patch("pathlib.Path.exists") patches the class method.
                # So we need to handle the instance.
                # Actually, simple lambda might be enough if we just want "True" for potential check.
                return True

            mock_exists.side_effect = lambda: True

            result = runner.train(Path("dataset.pckl.gzip"))

            assert result.status == JobStatus.COMPLETED
            assert result.validation_metrics["energy"] == 0.001

            # Verify input.yaml content
            input_yaml = runner.work_dir / "input.yaml"
            # Since we patched Path.exists globally, we can't trust it.
            # But 'open' works.
            # Wait, `write_file` tool writes real file. `runner.train` writes real file.
            # So `input.yaml` should be on disk in `tmp_path`.
            # But verify `open` wasn't mocked? No.

            with open(input_yaml) as f:
                data = yaml.safe_load(f)
                assert data["potential"]["elements"] == ["Si"]
                assert data["potential"]["embeddings"]["ALL"]["npot"] == "FinnisSinclair"

def test_active_set_selection(runner):
    with patch("subprocess.run") as mock_run:
        runner.select_active_set(Path("data.pckl.gzip"))
        # Check that pace_activeset was called
        args = mock_run.call_args[0][0]
        assert args[0] == "pace_activeset"

def test_active_set_failure(runner):
    # Should not crash, just log warning
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        runner.train(Path("data.pckl.gzip"))
        # Verify it proceeded to train even if active set failed (implied by no raise)

def test_train_failure(runner):
    with patch("subprocess.run") as mock_run:
        # Mock active set success, but train failure
        def side_effect(cmd, **kwargs):
            if "pace_train" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        result = runner.train(Path("data.pckl.gzip"))
        assert result.status == JobStatus.FAILED

def test_input_generation_hybrid(runner):
    """Verify ZBL repulsion block is added for hybrid potential."""
    dataset_path = Path("dataset.pckl.gzip")
    output_path = runner.work_dir / "input_test.yaml"

    runner._generate_input_yaml(dataset_path, output_path)

    with open(output_path) as f:
        data = yaml.safe_load(f)

    repulsion = data.get("potential", {}).get("repulsion")
    assert repulsion is not None
    assert repulsion["type"] == "zbl"
    # Defaults in PotentialConfig
    assert repulsion["inner_cutoff"] == 1.0
    assert repulsion["outer_cutoff"] == 2.0
