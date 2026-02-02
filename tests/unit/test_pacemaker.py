from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config import TrainingConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer


def test_pacemaker_train_command(temp_dir: Path) -> None:
    dataset = temp_dir / "data.pckl"
    dataset.touch()
    config = TrainingConfig(dataset_path=dataset, max_epochs=10, command="pace_train")
    trainer = PacemakerTrainer(config)

    with patch("subprocess.run") as mock_run:
        # Mock successful run
        mock_run.return_value.returncode = 0

        # We also need to mock output file existence check, or create it
        # Since logic checks "output.yace" in CWD, we can touch it.
        # But CWD might not be temp_dir.
        # We should patch Path.exists but specifically for output.yace
        # Easier to just let it run and mock existence.
        with patch("pathlib.Path.exists", return_value=True):
            trainer.train(dataset)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pace_train"
        assert "--dataset" in cmd
        assert "--max-epochs" in cmd
        assert "10" in cmd


def test_pacemaker_mock_mode(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")
    dataset = temp_dir / "data.pckl"
    dataset.touch()
    config = TrainingConfig(dataset_path=dataset)
    trainer = PacemakerTrainer(config)

    # We need to run in temp_dir so output.yace is created there
    monkeypatch.chdir(temp_dir)

    with patch("subprocess.run") as mock_run:
        output = trainer.train(dataset)
        mock_run.assert_not_called()
        assert output.name == "output.yace"
        assert output.exists()
